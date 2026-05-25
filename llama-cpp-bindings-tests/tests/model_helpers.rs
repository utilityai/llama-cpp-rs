#![expect(
    clippy::unnecessary_wraps,
    reason = "every trial returns anyhow::Result<()> to match the LlamaTestFn signature"
)]

use anyhow::Result;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn debug_format_includes_struct_name_and_model_field(fixture: &LlamaFixture<'_>) -> Result<()> {
    let formatted = format!("{:?}", fixture.model);

    assert!(formatted.contains("LlamaModel"));
    assert!(formatted.contains("model"));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn embedding_model_tool_call_markers_call_does_not_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let _markers = fixture.model.tool_call_markers();

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn embedding_model_streaming_markers_returns_ok_for_a_model_without_tool_calls(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let _markers = fixture.model.streaming_markers()?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn approximate_tok_env_is_cached_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture.model.approximate_tok_env();
    let second = fixture.model.approximate_tok_env();

    assert!(std::sync::Arc::ptr_eq(&first, &second));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128
)]
fn approximate_tok_env_falls_back_to_eos_when_eot_unavailable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let env = fixture.model.approximate_tok_env();
    let env_again = fixture.model.approximate_tok_env();

    assert!(
        std::sync::Arc::ptr_eq(&env, &env_again),
        "approximate_tok_env must return the same cached Arc for any model, including \
         the embedding model which lacks an EOT token (forcing the fallback-to-EOS path)"
    );

    Ok(())
}

llama_tests_main!();
