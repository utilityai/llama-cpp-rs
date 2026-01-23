//! Demonstrates streaming OpenAI-compatible deltas from a chat prompt with tools and JSON schema.
//!
//! Usage:
//!   cargo run --example openai_stream_tools_json_schema -- hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{
    AddBos, GrammarTriggerType, LlamaChatMessage, LlamaChatTemplate, LlamaModel, ToolDefinition,
};
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::json;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::PathBuf;

fn resolve_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let exe = args
        .next()
        .unwrap_or_else(|| "openai_stream_tools_json_schema".to_string());
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = resolve_model_path()?;
    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &params)?;

    let template = model
        .chat_template(None)
        .unwrap_or_else(|_| LlamaChatTemplate::new("chatml").expect("valid chat template"));

    let messages = vec![
        LlamaChatMessage::new("system".to_string(), "You are a tool caller.".to_string())?,
        LlamaChatMessage::new("user".to_string(), "Guess the weather in SF.".to_string())?,
    ];

    let response_schema = json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" },
            "summary": { "type": "string" },
            "unit": { "type": "string", "enum": ["c", "f"] }
        },
        "required": ["city", "summary"]
    });

    let result = model.apply_chat_template_with_tools(
        &template,
        &messages,
        None,
        Some(&response_schema),
        true,
    )?;

    println!("Prompt:\n{}", result.prompt);
    match result.grammar.as_deref() {
        Some(grammar) => println!("\nGrammar:\n{}", grammar),
        None => println!("\nGrammar: <none>"),
    }

    let tokens = model.str_to_token(&result.prompt, AddBos::Always)?;
    let n_predict: i32 = 128;
    let n_ctx = model
        .n_ctx_train()
        .max(tokens.len() as u32 + n_predict as u32);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last_index = tokens.len().saturating_sub(1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)?;

    let mut n_cur = batch.n_tokens();
    let max_tokens = n_cur + n_predict;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut preserved = HashSet::new();
    for token_str in &result.preserved_tokens {
        let tokens = model.str_to_token(token_str, AddBos::Never)?;
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
                        let tokens = model.str_to_token(&trigger.value, AddBos::Never)?;
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
                &model,
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
            match LlamaSampler::grammar(&model, grammar, "root") {
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

    let mut stream_parser = result.streaming_parser()?;
    let mut generated_text = String::new();
    let additional_stops = result.additional_stops.clone();

    println!("\nStreaming deltas:");
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
        let output_bytes = model.token_to_bytes(token, special)?;
        let mut output_string = String::with_capacity(32);
        let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        generated_text.push_str(&output_string);

        let stop_now = additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop));

        let update = stream_parser.update(&output_string, !stop_now)?;
        for delta in update.deltas {
            let delta_json = delta.to_oaicompat_value();
            println!("{}", serde_json::to_string(&delta_json)?);
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;

        if stop_now {
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

    let parsed = result.parse_response_oaicompat(&generated_text, false)?;
    let parsed_pretty = serde_json::to_string_pretty(&parsed)?;
    println!("\nFinal message:\n{}", parsed_pretty);

    Ok(())
}
