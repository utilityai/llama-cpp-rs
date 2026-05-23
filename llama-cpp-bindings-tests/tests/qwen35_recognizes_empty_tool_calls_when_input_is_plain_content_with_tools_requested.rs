use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

const TOOLS_JSON: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        }
    }
]"#;

const PLAIN_CONTENT: &str = "Sorry, I cannot help with that.";

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_recognizes_empty_tool_calls_when_input_is_plain_content_with_tools_requested(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, PLAIN_CONTENT, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "Qwen 3.5 with tools requested + plain content must produce Recognized (with empty \
             tool_calls); got Unrecognized"
        );
    };
    assert!(
        parsed.tool_calls.is_empty(),
        "expected no tool calls; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}

llama_tests_main!();
