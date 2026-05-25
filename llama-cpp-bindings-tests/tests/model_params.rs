#![expect(
    clippy::similar_names,
    reason = "model_path_str and model_path_cstr are both genuinely needed; renaming would not improve clarity"
)]

use std::ffi::CString;
use std::pin::pin;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::max_devices;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn fit_params_succeeds_with_test_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model_path_str = fixture
        .model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let model_path_cstr = CString::new(model_path_str)?;

    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    let result = params.as_mut().fit_params(
        &model_path_cstr,
        &mut context_params,
        &mut margins,
        512,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
    );

    let fit = result.map_err(|fit_error| anyhow::anyhow!("fit_params failed: {fit_error:?}"))?;
    assert!(fit.n_ctx > 0);

    Ok(())
}

llama_tests_main!();
