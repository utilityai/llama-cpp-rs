//! # `LLGuidance` Example
//!
//! This example demonstrates how to use the `LLGuidance` sampler for constrained decoding.
//!
//! ```console
//! cargo run --example llguidance --features llguidance -- <path_to_model>
//! ```

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use std::io::Write;

fn main() {
    let model_path = std::env::args().nth(1).expect("Please specify model path");
    let backend = LlamaBackend::init().unwrap();
    let params = LlamaModelParams::default();

    let model =
        LlamaModel::load_from_file(&backend, model_path, &params).expect("unable to load model");

    let prompt = "The weather in Paris is sunny and 22 degrees. Extract as JSON:\n";

    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("unable to create the llama_context");

    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .expect("failed to tokenize prompt");

    let mut batch = LlamaBatch::new(512, 1);
    let last_index = i32::try_from(tokens_list.len()).expect("prompt too long") - 1;
    for (i, token) in (0_i32..).zip(&tokens_list) {
        batch.add(*token, i, &[0], i == last_index).unwrap();
    }
    ctx.decode(&mut batch).expect("llama_decode() failed");

    // JSON Schema for weather
    let schema = r#"{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" }
  },
  "required": ["city", "temperature"]
}"#;

    // Build an LLGuidance sampler by getting the tok_env for our model (llama-cpp-2)
    // and use plain `llguidance` to build the grammar, parser -> matcher -> sampler. 
    let grammar = TopLevelGrammar::from_tagged_str("json", schema).expect("invalid grammar");
    let tok_env = LlamaSampler::llguidance_tok_env(&model);
    let factory = ParserFactory::new_simple(&tok_env).expect("failed to build parser factory");
    let parser = factory.create_parser(grammar).expect("failed to create parser");
    let matcher = Matcher::new(Ok(parser));

    let llg_sampler = LlamaSampler::from(matcher);

    // We must use a sampler chain that ends with a selector (like greedy or dist)
    let mut sampler = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    while n_cur <= 128 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if token == model.token_eos() {
            break;
        }

        let output_string = model
            .token_to_piece(token, &mut decoder, true, None)
            .unwrap();
        print!("{output_string}");
        std::io::stdout().flush().unwrap();

        sampler.accept(token);

        batch.clear();
        batch.add(token, n_cur, &[0], true).unwrap();
        n_cur += 1;
        ctx.decode(&mut batch).expect("failed to eval");
    }
    println!();
}
