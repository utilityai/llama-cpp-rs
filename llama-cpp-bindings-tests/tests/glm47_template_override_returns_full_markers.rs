#![expect(
    clippy::unnecessary_wraps,
    reason = "trial fns share the harness LlamaTestFn signature even when their bodies never propagate"
)]

use anyhow::Result;
use llama_cpp_bindings::ToolCallArgsShape;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn glm47_template_override_returns_full_markers(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let template = model
        .chat_template(None)
        .expect("GLM-4.7 chat template must be present");
    let template_str = template.to_str().expect("template must be valid UTF-8");
    assert!(template_str.contains("<arg_key>"));

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

llama_tests_main!();
