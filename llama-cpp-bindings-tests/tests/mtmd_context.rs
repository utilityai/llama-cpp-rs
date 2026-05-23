#![expect(
    clippy::unnecessary_wraps,
    reason = "trial fns share the harness LlamaTestFn signature even when their bodies never propagate"
)]

use anyhow::Result;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_and_supports_vision(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(mtmd_ctx.support_vision());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_with_null_byte_in_path_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_params = MtmdContextParams::default();
    let result = MtmdContext::init_from_file("path\0null", fixture.model, &mtmd_params);

    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn decode_use_mrope_is_true_for_qwen_vision(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        mtmd_ctx.decode_use_mrope(),
        "Qwen 3.5 / 3.6 mmproj uses mrope; decode_use_mrope must return true"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn support_audio_is_false_for_vision_only_mmproj(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        !mtmd_ctx.support_audio(),
        "Qwen 3.5 / 3.6 mmproj is vision-only; support_audio must return false"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn get_audio_sample_rate_is_none_for_vision_only_mmproj(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        mtmd_ctx.get_audio_sample_rate().is_none(),
        "Qwen 3.5 / 3.6 mmproj has no audio; get_audio_sample_rate must return None"
    );
    Ok(())
}

llama_tests_main!();
