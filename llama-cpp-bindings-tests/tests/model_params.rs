use std::ffi::CString;
use std::pin::pin;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::max_devices;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings_tests::FixtureSession;
use llama_cpp_bindings_tests::test_model;
use serial_test::serial;

#[test]
#[serial]
fn fit_params_succeeds_with_test_model() -> Result<()> {
    let _fixture = FixtureSession::open()?;

    let model_path = test_model::download_model()?;
    let model_path_str = model_path
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
