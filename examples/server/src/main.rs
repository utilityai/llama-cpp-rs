//! Minimal OpenAI-compatible chat completion server using Actix Web.
//!
//! Usage:
//!   cargo run -p openai-server -- <model_path>
//!   cargo run -p openai-server -- hf-model <repo> <model>
use actix_web::{http::StatusCode, web, App, HttpResponse, HttpServer};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, GrammarTriggerType, LlamaChatTemplate, LlamaModel, Special};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

struct AppState {
    backend: LlamaBackend,
    model: LlamaModel,
    default_template: Option<LlamaChatTemplate>,
    model_name: String,
}

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    message: String,
}

fn bad_request(message: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::BAD_REQUEST,
        message: message.into(),
    }
}

fn internal_error(message: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: message.into(),
    }
}

fn error_response(err: HttpError) -> HttpResponse {
    let body = json!({
        "error": {
            "message": err.message,
            "type": "invalid_request_error"
        }
    })
    .to_string();
    HttpResponse::build(err.status)
        .content_type("application/json")
        .body(body)
}

fn resolve_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let exe = args.next().unwrap_or_else(|| "openai_server".to_string());
    let first = args
        .next()
        .ok_or_else(|| format!("Usage: {exe} <model_path> | {exe} hf-model <repo> <model>"))?;

    if first == "hf-model" {
        let repo = args
            .next()
            .ok_or_else(|| "Missing Hugging Face repo".to_string())?;
        let model = args
            .next()
            .ok_or_else(|| "Missing model filename".to_string())?;
        let path = ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(repo)
            .get(&model)?;
        Ok(path)
    } else {
        Ok(PathBuf::from(first))
    }
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let mut anchored = String::new();
    if !pattern.starts_with('^') {
        anchored.push('^');
    }
    anchored.push_str(pattern);
    if !pattern.ends_with('$') {
        anchored.push('$');
    }
    anchored
}

fn parse_stop_sequences(request: &Value) -> Result<Vec<String>, HttpError> {
    let stop_value = match request.get("stop") {
        Some(value) => value,
        None => return Ok(Vec::new()),
    };

    match stop_value {
        Value::String(value) => Ok(vec![value.clone()]),
        Value::Array(values) => {
            let mut stops = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    Value::String(stop) => stops.push(stop.clone()),
                    _ => return Err(bad_request("stop array must contain strings")),
                }
            }
            Ok(stops)
        }
        Value::Null => Ok(Vec::new()),
        _ => Err(bad_request("stop must be a string or array of strings")),
    }
}

fn json_schema_value_to_string(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<String>, HttpError> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        Value::String(text) => Ok(Some(text.clone())),
        Value::Object(_) | Value::Array(_) => Ok(Some(value.to_string())),
        Value::Null => Ok(None),
        _ => Err(bad_request(format!("{label} must be a string or object"))),
    }
}

fn extract_json_schema(request: &Value) -> Result<Option<String>, HttpError> {
    if let Some(response_format) = request.get("response_format") {
        let format_type = response_format
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| bad_request("response_format.type must be a string"))?;
        if format_type == "json_schema" {
            let json_schema = response_format.get("json_schema").ok_or_else(|| {
                bad_request("response_format.json_schema is required for json_schema")
            })?;
            let schema_value = json_schema.get("schema").unwrap_or(json_schema);
            return json_schema_value_to_string(
                Some(schema_value),
                "response_format.json_schema.schema",
            );
        }
        return Ok(None);
    }

    json_schema_value_to_string(request.get("json_schema"), "json_schema")
}

fn build_sampler(
    model: &LlamaModel,
    result: &llama_cpp_2::model::ChatTemplateResult,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u32,
) -> Result<(LlamaSampler, HashSet<LlamaToken>), HttpError> {
    let mut preserved = HashSet::new();
    for token_str in &result.preserved_tokens {
        let tokens = model
            .str_to_token(token_str, AddBos::Never, true)
            .map_err(|e| internal_error(format!("preserved token error: {e}")))?;
        if tokens.len() == 1 {
            preserved.insert(tokens[0]);
        }
    }

    let grammar_sampler = if let Some(grammar) = result.grammar.as_deref() {
        if result.grammar_lazy {
            if result.grammar_triggers.is_empty() {
                return Err(internal_error(
                    "grammar_lazy enabled but no triggers were provided",
                ));
            }
            let mut trigger_patterns = Vec::new();
            let mut trigger_tokens = Vec::new();
            for trigger in &result.grammar_triggers {
                match trigger.trigger_type {
                    GrammarTriggerType::Token => {
                        if let Some(token) = trigger.token {
                            trigger_tokens.push(token);
                        }
                    }
                    GrammarTriggerType::Word => {
                        let tokens = model
                            .str_to_token(&trigger.value, AddBos::Never, true)
                            .map_err(|e| {
                                internal_error(format!("grammar trigger tokenize error: {e}"))
                            })?;
                        if tokens.len() == 1 {
                            if !preserved.contains(&tokens[0]) {
                                return Err(internal_error(format!(
                                    "grammar trigger word not preserved: {}",
                                    trigger.value
                                )));
                            }
                            trigger_tokens.push(tokens[0]);
                        } else {
                            trigger_patterns.push(regex_escape(&trigger.value));
                        }
                    }
                    GrammarTriggerType::Pattern => {
                        trigger_patterns.push(trigger.value.clone());
                    }
                    GrammarTriggerType::PatternFull => {
                        trigger_patterns.push(anchor_pattern(&trigger.value));
                    }
                }
            }

            Some(
                LlamaSampler::grammar_lazy_patterns(
                    model,
                    grammar,
                    "root",
                    &trigger_patterns,
                    &trigger_tokens,
                )
                .map_err(|e| internal_error(format!("grammar sampler error: {e}")))?,
            )
        } else {
            Some(
                LlamaSampler::grammar(model, grammar, "root")
                    .map_err(|e| internal_error(format!("grammar sampler error: {e}")))?,
            )
        }
    } else {
        None
    };

    let mut chain = Vec::new();
    if let Some(grammar) = grammar_sampler {
        chain.push(grammar);
    }

    if temperature > 0.0 {
        chain.push(LlamaSampler::temp(temperature));
        if top_k > 0 {
            chain.push(LlamaSampler::top_k(top_k));
        }
        if top_p < 1.0 {
            chain.push(LlamaSampler::top_p(top_p, 1));
        }
        chain.push(LlamaSampler::dist(seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }

    Ok((LlamaSampler::chain_simple(chain), preserved))
}

fn run_chat_completion(state: &AppState, body: &str) -> Result<String, HttpError> {
    let request: Value =
        serde_json::from_str(body).map_err(|e| bad_request(format!("invalid JSON: {e}")))?;

    let owned_template = match request.get("chat_template") {
        Some(Value::String(value)) => {
            let tmpl = LlamaChatTemplate::new(value)
                .map_err(|e| bad_request(format!("invalid chat_template: {e}")))?;
            Some(tmpl)
        }
        Some(Value::Null) | None => None,
        Some(_) => return Err(bad_request("chat_template must be a string or null")),
    };
    let template = owned_template
        .as_ref()
        .or(state.default_template.as_ref())
        .ok_or_else(|| bad_request("missing chat_template (model does not provide a template)"))?;

    let messages = request
        .get("messages")
        .ok_or_else(|| bad_request("missing messages"))?;
    if !messages.is_array() {
        return Err(bad_request("messages must be an array"));
    }
    let messages_json = messages.to_string();

    let tools_json = request.get("tools").map(|value| value.to_string());
    let tool_choice = match request.get("tool_choice") {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Null) | None => None,
        Some(_) => return Err(bad_request("tool_choice must be a string or null")),
    };

    let json_schema = extract_json_schema(&request)?;

    let grammar = match request.get("grammar") {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Null) | None => None,
        Some(_) => return Err(bad_request("grammar must be a string")),
    };

    let reasoning_format = match request.get("reasoning_format") {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Null) | None => None,
        Some(_) => return Err(bad_request("reasoning_format must be a string")),
    };

    let chat_template_kwargs = match request.get("chat_template_kwargs") {
        Some(Value::Object(_)) | Some(Value::Array(_)) => {
            Some(request["chat_template_kwargs"].to_string())
        }
        Some(Value::Null) | None => None,
        Some(_) => return Err(bad_request("chat_template_kwargs must be a JSON object")),
    };

    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if stream {
        return Err(bad_request("streaming is not supported in this example"));
    }

    let add_generation_prompt = request
        .get("add_generation_prompt")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let use_jinja = request
        .get("use_jinja")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let parallel_tool_calls = request
        .get("parallel_tool_calls")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let enable_thinking = request
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let add_bos = request
        .get("add_bos")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let add_eos = request
        .get("add_eos")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let max_tokens = request
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(1024) as u32;
    if max_tokens == 0 {
        return Err(bad_request("max_tokens must be greater than zero"));
    }

    let temperature = request
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    if temperature < 0.0 {
        return Err(bad_request("temperature must be >= 0"));
    }
    let top_p = request.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    if !(0.0 < top_p && top_p <= 1.0) {
        return Err(bad_request("top_p must be within (0, 1]"));
    }
    let top_k = request.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32;
    if top_k < 0 {
        return Err(bad_request("top_k must be >= 0"));
    }
    let seed = request.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32;

    let parse_tool_calls = tools_json.is_some()
        && tool_choice.as_deref() != Some("none")
        && json_schema.is_none()
        && grammar.is_none();

    let params = OpenAIChatTemplateParams {
        messages_json: messages_json.as_str(),
        tools_json: tools_json.as_deref(),
        tool_choice: tool_choice.as_deref(),
        json_schema: json_schema.as_deref(),
        grammar: grammar.as_deref(),
        reasoning_format: reasoning_format.as_deref(),
        chat_template_kwargs: chat_template_kwargs.as_deref(),
        add_generation_prompt,
        use_jinja,
        parallel_tool_calls,
        enable_thinking,
        add_bos,
        add_eos,
        parse_tool_calls,
    };

    let result = state
        .model
        .apply_chat_template_oaicompat(template, &params)
        .map_err(|e| internal_error(format!("apply_chat_template_oaicompat failed: {e}")))?;

    let tokens = state
        .model
        .str_to_token(&result.prompt, AddBos::Always, true)
        .map_err(|e| internal_error(format!("tokenization failed: {e}")))?;
    let n_ctx = state
        .model
        .n_ctx_train()
        .max(tokens.len() as u32 + max_tokens);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| internal_error(format!("context init failed: {e}")))?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last_index = tokens.len().saturating_sub(1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .map_err(|e| internal_error(format!("batch add failed: {e}")))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| internal_error(format!("decode failed: {e}")))?;

    let mut n_cur = batch.n_tokens();
    let max_tokens_total = n_cur + max_tokens as i32;
    let mut generated_text = String::new();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let (mut sampler, preserved) =
        build_sampler(&state.model, &result, temperature, top_p, top_k, seed)?;
    let mut additional_stops = result.additional_stops.clone();
    additional_stops.extend(parse_stop_sequences(&request)?);

    let mut finish_reason = "stop";
    while n_cur < max_tokens_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if state.model.is_eog_token(token) {
            break;
        }
        // sampler.accept(token);

        let special = if preserved.contains(&token) {
            Special::Tokenize
        } else {
            Special::Plaintext
        };
        let output_bytes = state
            .model
            .token_to_bytes(token, special)
            .map_err(|e| internal_error(format!("token decode failed: {e}")))?;
        let mut output_string = String::with_capacity(32);
        let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        generated_text.push_str(&output_string);
        completion_tokens += 1;

        if additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop))
        {
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| internal_error(format!("batch add failed: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode failed: {e}")))?;
    }

    if n_cur >= max_tokens_total {
        finish_reason = "length";
    }

    for stop in &additional_stops {
        if !stop.is_empty() && generated_text.ends_with(stop) {
            let new_len = generated_text.len().saturating_sub(stop.len());
            generated_text.truncate(new_len);
            break;
        }
    }

    let message_json = result
        .parse_response_oaicompat(&generated_text, false)
        .map_err(|e| internal_error(format!("parse response failed: {e}")))?;
    let message_value: Value = serde_json::from_str(&message_json)
        .map_err(|e| internal_error(format!("message JSON decode failed: {e}")))?;

    let model_name = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(state.model_name.as_str());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| internal_error(format!("time error: {e}")))?
        .as_secs();

    let response = json!({
        "id": format!("chatcmpl-{}", created),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message_value,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": tokens.len(),
            "completion_tokens": completion_tokens,
            "total_tokens": tokens.len() as u32 + completion_tokens
        }
    });

    Ok(response.to_string())
}

async fn chat_completions(state: web::Data<AppState>, body: web::Bytes) -> HttpResponse {
    let body_str = match std::str::from_utf8(&body) {
        Ok(value) => value,
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };

    match run_chat_completion(&state, body_str) {
        Ok(body) => HttpResponse::Ok()
            .content_type("application/json")
            .body(body),
        Err(err) => error_response(err),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path = resolve_model_path()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err.to_string()))?;
    let model_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("llama.cpp")
        .to_string();

    let backend = LlamaBackend::init()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &params)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;
    let default_template = model.chat_template(None).ok();

    let state = web::Data::new(AppState {
        backend,
        model,
        default_template,
        model_name,
    });

    let addr = std::env::var("LLAMA_RS_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    println!("OpenAI-compatible server listening on http://{addr}");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/v1/chat/completions", web::post().to(chat_completions))
    })
    .bind(addr)?
    .run()
    .await
}
