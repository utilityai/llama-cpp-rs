use anyhow::Result;
use llama_cpp_bindings::ToolCallArgsShape;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const GEMMA4_REPO: &str = "unsloth/gemma-4-E4B-it-GGUF";
const GEMMA4_FILE: &str = "gemma-4-E4B-it-Q4_K_M.gguf";

#[test]
fn gemma4_template_override_returns_full_markers() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(GEMMA4_REPO, GEMMA4_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let template = model
        .chat_template(None)
        .expect("Gemma 4 chat template must be present");
    let template_str = template.to_str().expect("template must be valid UTF-8");
    assert!(
        template_str.contains("<|tool_call>call:"),
        "Gemma 4 chat template must contain '<|tool_call>call:' fingerprint; \
         template starts with: {:?}",
        &template_str[..template_str.len().min(200)],
    );

    let markers = model
        .tool_call_markers()
        .expect("Gemma 4 must produce ToolCallMarkers via override registry");

    assert_eq!(markers.open, "<|tool_call>call:");
    assert_eq!(markers.close, "}");
    let ToolCallArgsShape::PairedQuote(shape) = markers.args_shape else {
        panic!("expected PairedQuote variant, got {:?}", markers.args_shape);
    };
    assert_eq!(shape.name_args_separator, "{");
    assert_eq!(shape.value_quote.open, "<|\"|>");
    assert_eq!(shape.value_quote.close, "<|\"|>");

    Ok(())
}
