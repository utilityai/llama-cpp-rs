use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArguments;
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

const GEMMA4_PAIRED_QUOTE_PAYLOAD: &str =
    "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma4_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome =
        fixture
            .model
            .parse_chat_message(TOOLS_JSON, GEMMA4_PAIRED_QUOTE_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for Gemma 4 PairedQuote on a Gemma-4 model; got Unrecognized");
    };
    assert_eq!(
        parsed.tool_calls.len(),
        1,
        "expected one tool call; got {:?}",
        parsed.tool_calls
    );
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    let location = match &parsed.tool_calls[0].arguments {
        ToolCallArguments::ValidJson(value) => value
            .get("location")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        ToolCallArguments::InvalidJson(raw) => {
            bail!("expected ValidJson, got InvalidJson: {raw}");
        }
    };
    assert_eq!(location.as_deref(), Some("Paris"));

    Ok(())
}

llama_tests_main!();
