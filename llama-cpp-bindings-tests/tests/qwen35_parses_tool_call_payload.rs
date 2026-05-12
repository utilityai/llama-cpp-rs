use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const QWEN35_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const QWEN35_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

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

fn load_qwen35() -> Result<(LlamaBackend, LlamaModel)> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;
    let path = download_file_from(QWEN35_REPO, QWEN35_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    Ok((backend, model))
}

#[test]
fn qwen35_parses_tool_call_payload() -> Result<()> {
    let (_backend, model) = load_qwen35()?;

    let outcome = model.parse_chat_message(TOOLS_JSON, QWEN_XML_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for Qwen XML on a Qwen-3.5 model; got Unrecognized");
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

#[test]
fn qwen35_parses_partial_tool_call_returns_pending_state() -> Result<()> {
    let (_backend, model) = load_qwen35()?;

    let outcome = model.parse_chat_message(TOOLS_JSON, PARTIAL_QWEN_XML_PAYLOAD, true)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for partial Qwen XML on a Qwen-3.5 model; got Unrecognized");
    };
    assert!(parsed.tool_calls.is_empty() || parsed.tool_calls.len() == 1);

    Ok(())
}

#[test]
fn qwen35_parses_multiple_tool_calls() -> Result<()> {
    let (_backend, model) = load_qwen35()?;

    let outcome = model.parse_chat_message(TOOLS_JSON, TWO_QWEN_XML_PAYLOADS, false)?;

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
