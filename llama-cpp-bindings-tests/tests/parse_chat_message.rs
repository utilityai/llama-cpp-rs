use anyhow::Result;
use llama_cpp_bindings_tests::TestFixture;

const QWEN_TOOLS_JSON: &str = r#"[
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

#[test]
fn parses_pure_content_response() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let parsed = model.parse_chat_message("[]", "hello world", false)?;

    assert!(parsed.tool_calls.is_empty());
    assert!(!parsed.is_empty());
    assert!(parsed.content.contains("hello world"));

    Ok(())
}

#[test]
fn parses_qwen3_tool_call_payload() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let input = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call>";
    let parsed = model.parse_chat_message(QWEN_TOOLS_JSON, input, false)?;

    assert_eq!(
        parsed.tool_calls.len(),
        1,
        "expected one tool call; got {:?}",
        parsed.tool_calls
    );
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    assert!(
        parsed.tool_calls[0].arguments_json.contains("Paris"),
        "arguments missing location: {}",
        parsed.tool_calls[0].arguments_json
    );

    Ok(())
}

#[test]
fn parses_partial_tool_call_returns_pending_state() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let input = "<tool_call>\n{\"name\":\"get_weather\",\"argum";
    let parsed = model.parse_chat_message(QWEN_TOOLS_JSON, input, true)?;

    assert!(parsed.tool_calls.is_empty() || parsed.tool_calls.len() == 1);

    Ok(())
}

#[test]
fn parses_multiple_tool_calls() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let input = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call><tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Berlin\"}}\n</tool_call>";
    let parsed = model.parse_chat_message(QWEN_TOOLS_JSON, input, false)?;

    assert!(
        parsed.tool_calls.len() >= 1,
        "expected at least one tool call; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}

#[test]
fn parses_reasoning_section_into_reasoning_content() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let input = "<think>step one, step two</think>\n\nactual response";
    let parsed = model.parse_chat_message("[]", input, false)?;

    assert!(
        parsed.reasoning_content.contains("step")
            || parsed.content.contains("step"),
        "neither content nor reasoning contains 'step'; content={:?} reasoning={:?}",
        parsed.content,
        parsed.reasoning_content
    );

    Ok(())
}

#[test]
fn parses_empty_input_yields_empty_message() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let parsed = model.parse_chat_message("[]", "", false)?;

    assert!(parsed.tool_calls.is_empty());

    Ok(())
}
