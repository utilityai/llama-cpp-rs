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
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma4_template_override_returns_full_markers(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
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

llama_tests_main!();
