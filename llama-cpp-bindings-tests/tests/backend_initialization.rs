use anyhow::Result;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    void_logs = true,
)]
fn void_logs_suppresses_output(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(
        fixture.model.n_vocab() > 0,
        "model must load successfully even when void_logs has been called before init"
    );

    Ok(())
}
