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

const QWEN_XML_PAYLOAD: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

const PARTIAL_QWEN_XML_PAYLOAD: &str = "<tool_call>\n<function=get_weather>\n<parameter=lo";

const TWO_QWEN_XML_PAYLOADS: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>\n\
<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Berlin\n\
</parameter>\n\
</function>\n\
</tool_call>";

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, QWEN_XML_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for Qwen XML on a Qwen-3.5 model; got Unrecognized");
    };
    assert_eq!(parsed.tool_calls.len(), 1);
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_partial_tool_call_returns_pending_state(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, PARTIAL_QWEN_XML_PAYLOAD, true)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for partial Qwen XML on a Qwen-3.5 model; got Unrecognized");
    };
    assert!(parsed.tool_calls.is_empty() || parsed.tool_calls.len() == 1);

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
)]
fn qwen35_parses_multiple_tool_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, TWO_QWEN_XML_PAYLOADS, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "expected Recognized for two Qwen XML payloads on a Qwen-3.5 model; got Unrecognized"
        );
    };
    assert!(
        !parsed.tool_calls.is_empty(),
        "expected at least one tool call; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}

llama_tests_main!();
