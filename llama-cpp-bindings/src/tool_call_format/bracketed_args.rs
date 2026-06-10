use llama_cpp_bindings_types::BracketedJsonShape;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_bindings_types::ToolCallMarkers;

use crate::error::BracketedArgsFailure;

enum ParseStep<'body> {
    Done,
    Call(ParsedToolCall, &'body str),
}

fn consume_optional_prefix<'body>(input: &'body str, literal: &str) -> &'body str {
    input.strip_prefix(literal).unwrap_or(input)
}

fn split_at_separator<'body>(
    input: &'body str,
    separator: &str,
) -> Option<(&'body str, &'body str)> {
    let (name_raw, after_separator) = input.split_once(separator)?;
    Some((name_raw, after_separator))
}

fn consume_one_json_value<'body>(
    input: &'body str,
    tool_name: &str,
) -> Result<(serde_json::Value, &'body str), BracketedArgsFailure> {
    let mut stream = serde_json::Deserializer::from_str(input).into_iter::<serde_json::Value>();
    let value = stream
        .next()
        .ok_or_else(|| BracketedArgsFailure::UnterminatedArguments {
            tool_name: tool_name.to_owned(),
        })?
        .map_err(|err| BracketedArgsFailure::InvalidJsonArguments {
            tool_name: tool_name.to_owned(),
            message: err.to_string(),
        })?;
    let consumed = stream.byte_offset();

    Ok((value, &input[consumed..]))
}

fn parse_one_call<'body>(
    input: &'body str,
    markers: &ToolCallMarkers,
    shape: &BracketedJsonShape,
) -> Result<ParseStep<'body>, BracketedArgsFailure> {
    if input.is_empty() {
        return Ok(ParseStep::Done);
    }

    let after_open = consume_optional_prefix(input, markers.open.as_str());

    let Some((name_raw, after_separator)) =
        split_at_separator(after_open, shape.name_args_separator.as_str())
    else {
        return Ok(ParseStep::Done);
    };

    let name = name_raw.trim().to_owned();
    if name.is_empty() {
        return Ok(ParseStep::Done);
    }

    let (arguments_value, after_arguments) = consume_one_json_value(after_separator, &name)?;

    let after_close = consume_optional_prefix(after_arguments, markers.close.as_str());

    Ok(ParseStep::Call(
        ParsedToolCall::new(
            String::new(),
            name,
            ToolCallArguments::ValidJson(arguments_value),
        ),
        after_close,
    ))
}

/// # Errors
///
/// Returns [`BracketedArgsFailure`] when the body looks like a bracketed-JSON
/// tool-call block (matches the name/args separator) but contains a structural
/// issue: invalid JSON arguments or a JSON value truncated mid-stream.
pub fn parse(
    body: &str,
    markers: &ToolCallMarkers,
    shape: &BracketedJsonShape,
) -> Result<Vec<ParsedToolCall>, BracketedArgsFailure> {
    if shape.name_args_separator.is_empty() {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    let mut remaining = body.trim_start();

    loop {
        match parse_one_call(remaining, markers, shape)? {
            ParseStep::Done => break,
            ParseStep::Call(call, rest) => {
                parsed.push(call);
                remaining = rest.trim_start();
            }
        }
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::BracketedJsonShape;
    use llama_cpp_bindings_types::ToolCallArgsShape;
    use llama_cpp_bindings_types::ToolCallArguments;
    use llama_cpp_bindings_types::ToolCallMarkers;
    use serde_json::json;

    use super::parse;
    use crate::error::BracketedArgsFailure;

    fn mistral3_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "[TOOL_CALLS]".to_owned(),
            close: String::new(),
            args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
            }),
        }
    }

    fn mistral3_shape() -> BracketedJsonShape {
        BracketedJsonShape {
            name_args_separator: "[ARGS]".to_owned(),
        }
    }

    #[test]
    fn parses_single_tool_call_with_open_marker_present() {
        let parsed = parse(
            "[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn parses_single_tool_call_when_classifier_stripped_open_marker() {
        let parsed = parse(
            "get_weather[ARGS]{\"location\":\"Paris\"}",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn parses_two_consecutive_tool_calls_with_repeated_open_marker() {
        let parsed = parse(
            "[TOOL_CALLS]a[ARGS]{\"x\":1}[TOOL_CALLS]b[ARGS]{\"y\":2}",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"x": 1}))
        );
        assert_eq!(parsed[1].name, "b");
        assert_eq!(
            parsed[1].arguments,
            ToolCallArguments::ValidJson(json!({"y": 2}))
        );
    }

    #[test]
    fn rejects_malformed_json_arguments_with_typed_failure() {
        let result = parse(
            "[TOOL_CALLS]get_weather[ARGS]{\"location\":}",
            &mistral3_markers(),
            &mistral3_shape(),
        );

        let failure = result.expect_err("malformed JSON must produce a typed failure");

        assert_eq!(
            std::mem::discriminant(&failure),
            std::mem::discriminant(&BracketedArgsFailure::InvalidJsonArguments {
                tool_name: String::new(),
                message: String::new(),
            }),
        );
    }

    #[test]
    fn rejects_truncated_json_arguments_with_unterminated_failure() {
        let failure = parse(
            "[TOOL_CALLS]get_weather[ARGS]   ",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect_err("truncated arguments must produce a typed failure");

        assert_eq!(
            failure,
            BracketedArgsFailure::UnterminatedArguments {
                tool_name: "get_weather".to_owned(),
            },
        );
    }

    #[test]
    fn returns_empty_vec_for_separator_with_only_whitespace_name() {
        let parsed = parse(
            "[TOOL_CALLS]   [ARGS]{\"x\":1}",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect("whitespace-name input must parse");

        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_vec_when_shape_has_empty_separator() {
        let mut shape = mistral3_shape();
        shape.name_args_separator.clear();
        let parsed = parse(
            "[TOOL_CALLS]get_weather[ARGS]{\"x\":1}",
            &mistral3_markers(),
            &shape,
        )
        .expect("empty-separator shape must parse");

        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_vec_for_empty_body() {
        let parsed =
            parse("", &mistral3_markers(), &mistral3_shape()).expect("empty body must parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_vec_when_body_lacks_separator() {
        let parsed = parse(
            "plain text without separator",
            &mistral3_markers(),
            &mistral3_shape(),
        )
        .expect("body without separator must parse");
        assert!(parsed.is_empty());
    }
}
