//! Demonstrates tool calling with explicit reasoning parsing enabled.
//!
//! Usage:
//!   cargo run --example tools_reasoning -- [--continous] hf-model unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::ChatTemplateResult;
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{AddBos, GrammarTriggerType, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::json;
use std::collections::HashSet;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

fn resolve_args() -> Result<(PathBuf, bool), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let exe = args.next().unwrap_or_else(|| "tools_reasoning".to_string());
    let mut continuous = false;
    let mut positional = Vec::new();
    for arg in args {
        if arg == "--continous" {
            continuous = true;
        } else {
            positional.push(arg);
        }
    }

    let mut positional = positional.into_iter();
    let first = positional.next().ok_or_else(|| {
        format!(
            "Usage: {exe} [--continous] <model_path> | {exe} [--continous] hf-model <repo> <model>"
        )
    })?;

    if first == "hf-model" {
        let repo = positional
            .next()
            .ok_or_else(|| "Missing Hugging Face repo".to_string())?;
        let model = positional
            .next()
            .ok_or_else(|| "Missing model filename".to_string())?;
        let path = ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(repo)
            .get(&model)?;
        Ok((path, continuous))
    } else {
        Ok((PathBuf::from(first), continuous))
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

fn generate_text(
    model: &LlamaModel,
    backend: &LlamaBackend,
    result: &ChatTemplateResult,
    n_predict: i32,
) -> String {
    println!("Prompt:\n{}", result.prompt);
    println!("thinking_forced_open: {}", result.thinking_forced_open);
    match result.grammar.as_deref() {
        Some(grammar) => println!("\nGrammar:\n{}", grammar),
        None => println!("\nGrammar: <none>"),
    }

    let tokens = model
        .str_to_token(&result.prompt, AddBos::Always)
        .expect("Failed to tokenize prompt");
    let n_ctx = model
        .n_ctx_train()
        .max(tokens.len() as u32 + n_predict as u32);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("Failed to create context");

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last_index = tokens.len().saturating_sub(1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .expect("Failed to add token");
    }

    ctx.decode(&mut batch).expect("Initial decode failed");

    let mut n_cur = batch.n_tokens();
    let max_tokens = n_cur + n_predict;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut preserved = HashSet::new();
    for token_str in &result.preserved_tokens {
        let tokens = model
            .str_to_token(token_str, AddBos::Never)
            .expect("Failed to tokenize preserved token");
        if tokens.len() == 1 {
            preserved.insert(tokens[0]);
        }
    }

    let mut sampler = if let Some(grammar) = result.grammar.as_deref() {
        if result.grammar_lazy {
            if result.grammar_triggers.is_empty() {
                panic!("grammar_lazy enabled but no triggers provided");
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
                            .str_to_token(&trigger.value, AddBos::Never)
                            .expect("Failed to tokenize trigger word");
                        if tokens.len() == 1 {
                            if !preserved.contains(&tokens[0]) {
                                panic!(
                                    "Grammar trigger word should be marked as preserved token: {}",
                                    trigger.value
                                );
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

            match LlamaSampler::grammar_lazy_patterns(
                model,
                grammar,
                "root",
                &trigger_patterns,
                &trigger_tokens,
            ) {
                Ok(grammar_sampler) => {
                    println!("Applying Lazy Grammar sampler chain");
                    LlamaSampler::chain_simple([grammar_sampler, LlamaSampler::greedy()])
                }
                Err(err) => {
                    eprintln!("Failed to init lazy grammar sampler: {err}");
                    LlamaSampler::greedy()
                }
            }
        } else {
            match LlamaSampler::grammar(model, grammar, "root") {
                Ok(grammar_sampler) => {
                    println!("Applying Grammar sampler chain");
                    LlamaSampler::chain_simple([grammar_sampler, LlamaSampler::greedy()])
                }
                Err(err) => {
                    eprintln!("Failed to init grammar sampler: {err}");
                    LlamaSampler::greedy()
                }
            }
        }
    } else {
        LlamaSampler::greedy()
    };

    let mut generated_text = String::new();
    let additional_stops = result.additional_stops.clone();

    while n_cur <= max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if model.is_eog_token(token) {
            break;
        }

        let special = if preserved.contains(&token) {
            Special::Tokenize
        } else {
            Special::Plaintext
        };
        let output_bytes = model
            .token_to_bytes(token, special)
            .expect("Failed to decode token");
        let mut output_string = String::with_capacity(32);
        let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        generated_text.push_str(&output_string);
        print!("{output_string}");
        std::io::stdout().flush().expect("stdout flush failed");

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .expect("Failed to add token");
        n_cur += 1;
        ctx.decode(&mut batch).expect("Decode failed");

        if additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop))
        {
            break;
        }
    }

    for stop in &additional_stops {
        if !stop.is_empty() && generated_text.ends_with(stop) {
            let new_len = generated_text.len().saturating_sub(stop.len());
            generated_text.truncate(new_len);
            break;
        }
    }

    generated_text
}

fn mock_tool_response(function_name: &str, arguments: &serde_json::Value) -> String {
    match function_name {
        "get_weather" => {
            let city = arguments
                .get("city")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("Unknown City");
            let unit = arguments
                .get("unit")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("c");
            match unit {
                "f" => format!("Mock weather for {city}: sunny, 64 F, light wind."),
                _ => format!("Mock weather for {city}: sunny, 18 C, light wind."),
            }
        }
        _ => format!("Mock response for tool `{function_name}` with args: {arguments}"),
    }
}

fn assistant_tool_call_message(parsed_message: &serde_json::Value) -> serde_json::Value {
    let tool_calls = parsed_message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .map(|tool_call| {
            let arguments = tool_call
                .get("function")
                .and_then(|function| function.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            json!({
                "id": tool_call.get("id").cloned().unwrap_or(serde_json::Value::Null),
                "type": tool_call.get("type").cloned().unwrap_or_else(|| json!("function")),
                "function": {
                    "name": tool_call
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                    "arguments": arguments.to_string(),
                }
            })
        })
        .collect::<Vec<_>>();

    json!({
        "role": parsed_message.get("role").cloned().unwrap_or_else(|| json!("assistant")),
        "content": parsed_message.get("content").cloned().unwrap_or(serde_json::Value::Null),
        "reasoning_content": parsed_message
            .get("reasoning_content")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "tool_calls": tool_calls,
    })
}

fn main() {
    let (model_path, continuous) = resolve_args().unwrap_or_else(|err| panic!("{err}"));
    let backend = LlamaBackend::init().expect("Failed to init backend");
    let params = LlamaModelParams::default();
    let model =
        LlamaModel::load_from_file(&backend, model_path, &params).expect("Failed to load model");

    let template = model
        .chat_template(None)
        .unwrap_or_else(|_| LlamaChatTemplate::new("chatml").expect("valid chat template"));

    let messages = json!([
        {
            "role": "system",
            "content": "You are a tool caller. Think step by step, then call tools when needed."
        },
        {
            "role": "user",
            "content": "Get the weather in Paris and summarize it."
        }
    ]);

    let tools_json = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch current weather by city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name." },
                        "unit": { "type": "string", "enum": ["c", "f"] }
                    },
                    "required": ["city"]
                }
            }
        }
    ])
    .to_string();

    let messages_json = messages.to_string();
    let options = OpenAIChatTemplateParams {
        messages_json: &messages_json,
        tools_json: Some(&tools_json),
        tool_choice: Some("auto"),
        json_schema: None,
        grammar: None,
        reasoning_format: Some("auto"),
        chat_template_kwargs: Some("{}"),
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: false,
        enable_thinking: true,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: true,
    };

    let result = model
        .apply_chat_template_oaicompat(&template, &options)
        .expect("Failed to apply chat template");

    let generated_text = generate_text(&model, &backend, &result, 128);
    let mut parsed_summaries = Vec::new();

    let parsed_json = result
        .parse_response_oaicompat(&generated_text, false)
        .expect("Failed to parse raw response");
    let parsed_message: serde_json::Value =
        serde_json::from_str(&parsed_json).expect("Failed to decode parsed response");

    println!("\n\nRaw parsed message:\n{}", parsed_json);
    let parsed_pretty =
        serde_json::to_string_pretty(&parsed_message).expect("Failed to format parsed response");
    println!("\n\nPretty parsed message:\n{}", parsed_pretty);
    parsed_summaries.push(("Initial raw parsed message", parsed_json.clone()));
    parsed_summaries.push(("Initial pretty parsed message", parsed_pretty));

    if continuous {
        let tool_calls = parsed_message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            println!("\nNo tool calls were produced, skipping --continous follow-up.");
        } else {
            let mut conversation = messages.as_array().cloned().expect("messages is an array");
            conversation.push(assistant_tool_call_message(&parsed_message));

            for tool_call in &tool_calls {
                let tool_call_id = tool_call
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("mock_tool_call_id");
                let function = tool_call
                    .get("function")
                    .expect("tool call should include function");
                let function_name = function
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown_tool");
                let function_arguments = function
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_output = mock_tool_response(function_name, &function_arguments);
                println!(
                    "\nMock tool response for {}({}):\n{}",
                    function_name, function_arguments, tool_output
                );
                conversation.push(json!({
                    "role": "tool",
                    "name": function_name,
                    "content": tool_output,
                    "tool_call_id": tool_call_id,
                }));
            }

            let follow_up_messages = serde_json::Value::Array(conversation);
            let follow_up_json = follow_up_messages.to_string();
            let follow_up_options = OpenAIChatTemplateParams {
                messages_json: &follow_up_json,
                tools_json: Some(&tools_json),
                tool_choice: Some("none"),
                json_schema: None,
                grammar: None,
                reasoning_format: Some("auto"),
                chat_template_kwargs: Some("{}"),
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: false,
                enable_thinking: true,
                add_bos: false,
                add_eos: false,
                parse_tool_calls: false,
            };

            println!("\n\n--- Follow-up generation with mock tool output ---\n");
            let follow_up_result = model
                .apply_chat_template_oaicompat(&template, &follow_up_options)
                .expect("Failed to apply follow-up chat template");
            let follow_up_text = generate_text(&model, &backend, &follow_up_result, 192);

            let follow_up_raw = follow_up_result
                .parse_response_oaicompat(&follow_up_text, false)
                .expect("Failed to parse follow-up raw response");
            let follow_up_typed: serde_json::Value =
                serde_json::from_str(&follow_up_raw).expect("Failed to decode follow-up response");
            let follow_up_typed_pretty = serde_json::to_string_pretty(&follow_up_typed)
                .expect("Failed to format follow-up response");
            println!("\n\nFollow-up raw parsed message:\n{}", follow_up_raw);
            println!(
                "\n\nFollow-up pretty parsed message:\n{}",
                follow_up_typed_pretty
            );
            parsed_summaries.push(("Follow-up raw parsed message", follow_up_raw));
            parsed_summaries.push(("Follow-up pretty parsed message", follow_up_typed_pretty));
        }
    }

    println!("\n\n=== Parsed Message Summary ===");
    for (label, message) in parsed_summaries {
        println!("\n{label}:\n{message}");
    }
}
