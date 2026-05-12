use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings_tests::FixtureSession;

#[test]
fn parses_pure_content_response() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let outcome = model.parse_chat_message("[]", "hello world", false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for plain content; got Unrecognized");
    };
    assert!(parsed.tool_calls.is_empty());
    assert!(!parsed.is_empty());
    assert!(parsed.content.contains("hello world"));

    Ok(())
}

#[test]
fn parses_reasoning_section_into_reasoning_content() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let input = "<think>step one, step two</think>\n\nactual response";
    let outcome = model.parse_chat_message("[]", input, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for reasoning section; got Unrecognized");
    };
    assert!(
        parsed.reasoning_content.contains("step") || parsed.content.contains("step"),
        "neither content nor reasoning contains 'step'; content={:?} reasoning={:?}",
        parsed.content,
        parsed.reasoning_content
    );

    Ok(())
}

#[test]
fn parses_empty_input_yields_empty_message() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let outcome = model.parse_chat_message("[]", "", false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for empty input; got Unrecognized");
    };
    assert!(parsed.tool_calls.is_empty());

    Ok(())
}

#[test]
fn parses_malformed_tools_json_returns_tools_json_invalid_error() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let result = model.parse_chat_message("not_a_json[}", "hello", false);

    assert!(matches!(
        result,
        Err(llama_cpp_bindings::ParseChatMessageError::ToolsJsonInvalid(
            _
        ))
    ));
}

#[test]
fn parses_non_array_tools_json_returns_tools_json_not_array_error() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let result = model.parse_chat_message("{\"foo\": 1}", "hello", false);

    assert!(matches!(
        result,
        Err(llama_cpp_bindings::ParseChatMessageError::ToolsJsonNotArray)
    ));
}

#[test]
fn parses_with_tools_null_byte_returns_tools_json_invalid_error() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let result = model.parse_chat_message("[]\0extra", "hello", false);

    assert!(matches!(
        result,
        Err(llama_cpp_bindings::ParseChatMessageError::ToolsJsonInvalid(
            _
        ))
    ));
}

#[test]
fn parses_with_input_null_byte_returns_tools_serialization_error() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let result = model.parse_chat_message("[]", "hello\0world", false);

    assert!(matches!(
        result,
        Err(llama_cpp_bindings::ParseChatMessageError::ToolsSerialization(_))
    ));
}
