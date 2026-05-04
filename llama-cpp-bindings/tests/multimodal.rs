#![cfg(feature = "tests_that_use_llms")]

use std::num::NonZeroU32;

use anyhow::{Context, Result};
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_bindings::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::test_model;

#[test]
fn multimodal_vision_inference_produces_output() -> Result<()> {
    let model_path = test_model::download_model()?;
    let mmproj_path = test_model::download_mmproj()?;

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let n_ctx = NonZeroU32::new(4096);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(n_ctx)
        .with_n_batch(512);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create llama context")?;

    let mtmd_params = MtmdContextParams::default();
    let mmproj_path_str = mmproj_path
        .to_str()
        .with_context(|| "mmproj path is not valid UTF-8")?;
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_path_str, &model, &mtmd_params)
        .with_context(|| "unable to create mtmd context")?;

    assert!(
        mtmd_ctx.support_vision(),
        "model should support vision input"
    );

    let image_path = test_model::fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, image_path_str)
        .with_context(|| "failed to load image from file")?;

    let marker = llama_cpp_bindings::mtmd::mtmd_default_marker();
    let user_content = format!("{marker}What animals do you see in this image?");
    let chat_template = model.chat_template(None)?;
    let messages = [LlamaChatMessage::new("user".to_string(), user_content)?];
    let formatted_prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let input_text = MtmdInputText {
        text: formatted_prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let total_tokens = chunks.total_tokens();
    eprintln!(
        "tokenized into {} chunks, {} total tokens",
        chunks.len(),
        total_tokens
    );

    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)
        .with_context(|| "failed to evaluate chunks")?;

    eprintln!("evaluated chunks, n_past = {n_past}");

    let mut sampler = LlamaSampler::greedy();
    let mut generated = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let max_tokens = 512;

    let mut batch = llama_cpp_bindings::llama_batch::LlamaBatch::new(512, 1)?;
    let mut current_position = n_past;

    for _ in 0..max_tokens {
        let token = sampler.sample(&ctx, -1)?;

        if model.is_eog_token(token) {
            break;
        }

        let output_string = model
            .token_to_piece(token, &mut decoder, false, None)
            .with_context(|| "failed to convert token to piece")?;
        generated.push_str(&output_string);

        batch.clear();
        batch.add(token, current_position, &[0], true)?;
        current_position += 1;

        ctx.decode(&mut batch)
            .with_context(|| "failed to decode generated token")?;
    }

    eprintln!("generated text: {generated}");

    assert!(
        !generated.is_empty(),
        "model should generate at least one token from image input"
    );

    Ok(())
}
