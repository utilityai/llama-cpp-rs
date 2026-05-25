use std::time::Duration;

use anyhow::{Context, Result};
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::ggml_time_us;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |accumulator, &value| value.mul_add(value, accumulator))
        .sqrt();

    input.iter().map(|&value| value / magnitude).collect()
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_threads_batch = 8,
    embeddings = true,
)]
fn embedding_generation_produces_vectors(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;

    let mut ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let n_ctx = usize::try_from(ctx.n_ctx())?;
    assert!(tokens.len() <= n_ctx, "prompt exceeds context window size");

    let t_main_start = ggml_time_us();

    let mut classifier = model.sampled_token_classifier();
    let mut batch = LlamaBatch::new(n_ctx, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "failed to get embeddings")?;
    let normalized = normalize(embedding);

    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

    eprintln!(
        "created embedding with {} dimensions in {:.2} s",
        normalized.len(),
        duration.as_secs_f32()
    );

    assert!(
        !normalized.is_empty(),
        "embedding should have at least one dimension"
    );

    let magnitude: f32 = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "normalized embedding magnitude should be approximately 1.0, got {magnitude}"
    );

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, prompt_token_count);
    assert_eq!(usage.completion_tokens(), 0);

    Ok(())
}

llama_tests_main!();
