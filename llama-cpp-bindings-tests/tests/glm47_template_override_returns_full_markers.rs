use anyhow::Result;
use llama_cpp_bindings::ToolCallArgsShape;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const GLM47_REPO: &str = "unsloth/GLM-4.7-Flash-GGUF";
const GLM47_FILE: &str = "GLM-4.7-Flash-Q4_K_M.gguf";

#[test]
fn glm47_template_override_returns_full_markers() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(GLM47_REPO, GLM47_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let template = model
        .chat_template(None)
        .expect("GLM-4.7 chat template must be present");
    let template_str = template.to_str().expect("template must be valid UTF-8");
    assert!(
        template_str.contains("<arg_key>"),
        "GLM-4.7 chat template must contain '<arg_key>' fingerprint; \
         template starts with: {:?}",
        &template_str[..template_str.len().min(200)],
    );

    let markers = model
        .tool_call_markers()
        .expect("GLM-4.7 must produce ToolCallMarkers via override registry");

    assert_eq!(markers.open, "<tool_call>");
    assert_eq!(markers.close, "</tool_call>");
    let ToolCallArgsShape::KeyValueXmlTags(shape) = markers.args_shape else {
        panic!(
            "expected KeyValueXmlTags variant, got {:?}",
            markers.args_shape
        );
    };
    assert_eq!(shape.key_open, "<arg_key>");
    assert_eq!(shape.key_close, "</arg_key>");
    assert_eq!(shape.value_open, "<arg_value>");
    assert_eq!(shape.value_close, "</arg_value>");

    Ok(())
}
