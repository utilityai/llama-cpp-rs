use anyhow::Result;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::test_model;
use serial_test::serial;

#[test]
#[serial]
fn void_logs_suppresses_output() -> Result<()> {
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    let model_path = test_model::download_model()?;
    let model_params = inference_model_params();
    let _model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    Ok(())
}
