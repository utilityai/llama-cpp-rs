use llama_cpp_bindings_types::KeyValueXmlTagsShape;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_bindings_types::ToolCallMarkers;
use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_until;

use crate::error::KeyValueXmlTagsFailure;

enum ParseStep<'body> {
    Done,
    Call(ParsedToolCall, &'body str),
}

const fn shape_is_complete(shape: &KeyValueXmlTagsShape) -> bool {
    !shape.key_open.is_empty()
        && !shape.key_close.is_empty()
        && !shape.value_open.is_empty()
        && !shape.value_close.is_empty()
}

fn skip_to_next_open<'body>(input: &'body str, open: &str) -> Option<&'body str> {
    let take_result: IResult<&'body str, &'body str> = take_until(open).parse(input);
    let (after_prefix_inclusive, _) = take_result.ok()?;
    let consume_result: IResult<&'body str, &'body str> =
        tag(open).parse(after_prefix_inclusive);
    let (after_open, _) = consume_result.ok()?;

    Some(after_open)
}

fn parameter_value_to_json(raw: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .unwrap_or_else(|| serde_json::Value::String(raw.to_owned()))
}

fn parse_one_parameter<'body>(
    input: &'body str,
    shape: &KeyValueXmlTagsShape,
    function_name: &str,
) -> Result<Option<(String, serde_json::Value, &'body str)>, KeyValueXmlTagsFailure> {
    let take_result: IResult<&'body str, &'body str> =
        take_until(shape.key_open.as_str()).parse(input);
    let Ok((after_key_open_inclusive, _)) = take_result else {
        return Ok(None);
    };
    let consume_result: IResult<&'body str, &'body str> =
        tag(shape.key_open.as_str()).parse(after_key_open_inclusive);
    let Ok((after_key_open, _)) = consume_result else {
        return Ok(None);
    };

    let key_close_position = after_key_open.find(shape.key_close.as_str()).ok_or_else(|| {
        KeyValueXmlTagsFailure::UnclosedKeyTag {
            function_name: function_name.to_owned(),
            expected_close: shape.key_close.clone(),
        }
    })?;
    let key = after_key_open[..key_close_position].trim().to_owned();
    if key.is_empty() {
        return Err(KeyValueXmlTagsFailure::EmptyKey {
            function_name: function_name.to_owned(),
        });
    }
    let after_key_close = &after_key_open[key_close_position + shape.key_close.len()..];

    let value_open_take: IResult<&str, &str> =
        take_until(shape.value_open.as_str()).parse(after_key_close);
    let Ok((after_value_open_inclusive, _)) = value_open_take else {
        return Err(KeyValueXmlTagsFailure::MissingValueTag {
            function_name: function_name.to_owned(),
            key,
            expected_open: shape.value_open.clone(),
        });
    };
    let value_open_consume: IResult<&str, &str> =
        tag(shape.value_open.as_str()).parse(after_value_open_inclusive);
    let Ok((after_value_open, _)) = value_open_consume else {
        return Err(KeyValueXmlTagsFailure::MissingValueTag {
            function_name: function_name.to_owned(),
            key,
            expected_open: shape.value_open.clone(),
        });
    };

    let value_close_position =
        after_value_open
            .find(shape.value_close.as_str())
            .ok_or_else(|| KeyValueXmlTagsFailure::UnclosedValueTag {
                function_name: function_name.to_owned(),
                key: key.clone(),
                expected_close: shape.value_close.clone(),
            })?;
    let raw_value = &after_value_open[..value_close_position];
    let value = parameter_value_to_json(raw_value);
    let after_value_close = &after_value_open[value_close_position + shape.value_close.len()..];

    Ok(Some((key, value, after_value_close)))
}

fn collect_parameters(
    function_body: &str,
    shape: &KeyValueXmlTagsShape,
    function_name: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, KeyValueXmlTagsFailure> {
    let mut parameters = serde_json::Map::new();
    let mut remaining = function_body;

    while let Some((key, value, rest)) = parse_one_parameter(remaining, shape, function_name)? {
        parameters.insert(key, value);
        remaining = rest;
    }

    Ok(parameters)
}

fn parse_one_call<'body>(
    input: &'body str,
    markers: &ToolCallMarkers,
    shape: &KeyValueXmlTagsShape,
) -> Result<ParseStep<'body>, KeyValueXmlTagsFailure> {
    let Some(after_open) = skip_to_next_open(input, &markers.open) else {
        return Ok(ParseStep::Done);
    };

    let Some(close_position) = after_open.find(markers.close.as_str()) else {
        return Err(KeyValueXmlTagsFailure::UnclosedFunctionBlock {
            expected_close: markers.close.clone(),
        });
    };
    let function_block = &after_open[..close_position];
    let after_function_close = &after_open[close_position + markers.close.len()..];

    let (name_end, has_args) = function_block
        .find(shape.key_open.as_str())
        .map_or((function_block.len(), false), |position| (position, true));
    let function_name = function_block[..name_end].trim().to_owned();
    if function_name.is_empty() {
        return Err(KeyValueXmlTagsFailure::EmptyFunctionName);
    }

    let args_section = if has_args {
        &function_block[name_end..]
    } else {
        ""
    };
    let arguments_object = collect_parameters(args_section, shape, &function_name)?;
    let arguments_value = serde_json::Value::Object(arguments_object);
    let arguments = ToolCallArguments::from_string(arguments_value.to_string());

    Ok(ParseStep::Call(
        ParsedToolCall::new(String::new(), function_name, arguments),
        after_function_close,
    ))
}

/// # Errors
///
/// Returns [`KeyValueXmlTagsFailure`] when the body looks like a key-value-XML
/// tool-call block (matches the open marker) but contains a structural issue:
/// empty function/key name, missing key/value tag, or unclosed function block.
pub fn parse(
    body: &str,
    markers: &ToolCallMarkers,
    shape: &KeyValueXmlTagsShape,
) -> Result<Vec<ParsedToolCall>, KeyValueXmlTagsFailure> {
    if !shape_is_complete(shape) || markers.open.is_empty() || markers.close.is_empty() {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    let mut remaining = body;

    loop {
        match parse_one_call(remaining, markers, shape)? {
            ParseStep::Done => break,
            ParseStep::Call(call, rest) => {
                parsed.push(call);
                remaining = rest;
            }
        }
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::KeyValueXmlTagsShape;
    use llama_cpp_bindings_types::ToolCallArgsShape;
    use llama_cpp_bindings_types::ToolCallArguments;
    use llama_cpp_bindings_types::ToolCallMarkers;
    use serde_json::json;

    use super::parse;
    use crate::error::KeyValueXmlTagsFailure;

    fn glm47_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::KeyValueXmlTags(glm47_shape()),
        }
    }

    fn glm47_shape() -> KeyValueXmlTagsShape {
        KeyValueXmlTagsShape {
            key_open: "<arg_key>".to_owned(),
            key_close: "</arg_key>".to_owned(),
            value_open: "<arg_value>".to_owned(),
            value_close: "</arg_value>".to_owned(),
        }
    }

    #[test]
    fn parses_single_call_with_one_argument() {
        let body = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";
        let parsed = parse(body, &glm47_markers(), &glm47_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn parses_call_with_multiple_arguments() {
        let body = "<tool_call>set_thermostat<arg_key>room</arg_key><arg_value>kitchen</arg_value><arg_key>celsius</arg_key><arg_value>21</arg_value></tool_call>";
        let parsed = parse(body, &glm47_markers(), &glm47_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "set_thermostat");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"room": "kitchen", "celsius": 21})),
        );
    }

    #[test]
    fn parses_two_calls_in_one_body() {
        let body = "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call><tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>";
        let parsed = parse(body, &glm47_markers(), &glm47_shape()).expect("must parse");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert_eq!(parsed[1].name, "b");
    }

    #[test]
    fn parses_call_with_no_arguments() {
        let body = "<tool_call>ping</tool_call>";
        let parsed = parse(body, &glm47_markers(), &glm47_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "ping");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({})),
        );
    }

    #[test]
    fn rejects_unclosed_function_block_with_typed_failure() {
        let body = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value>";
        let result = parse(body, &glm47_markers(), &glm47_shape());

        match result.expect_err("must error") {
            KeyValueXmlTagsFailure::UnclosedFunctionBlock { expected_close } => {
                assert_eq!(expected_close, "</tool_call>");
            }
            other => panic!("expected UnclosedFunctionBlock, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_function_name_with_typed_failure() {
        let body = "<tool_call><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>";
        let result = parse(body, &glm47_markers(), &glm47_shape());

        match result.expect_err("must error") {
            KeyValueXmlTagsFailure::EmptyFunctionName => {}
            other => panic!("expected EmptyFunctionName, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unclosed_key_tag_with_typed_failure() {
        let body = "<tool_call>f<arg_key>location</tool_call>";
        let result = parse(body, &glm47_markers(), &glm47_shape());

        match result.expect_err("must error") {
            KeyValueXmlTagsFailure::UnclosedKeyTag { function_name, .. } => {
                assert_eq!(function_name, "f");
            }
            other => panic!("expected UnclosedKeyTag, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_value_tag_with_typed_failure() {
        let body = "<tool_call>f<arg_key>location</arg_key>Paris</tool_call>";
        let result = parse(body, &glm47_markers(), &glm47_shape());

        match result.expect_err("must error") {
            KeyValueXmlTagsFailure::MissingValueTag {
                function_name,
                key,
                ..
            } => {
                assert_eq!(function_name, "f");
                assert_eq!(key, "location");
            }
            other => panic!("expected MissingValueTag, got {other:?}"),
        }
    }

    #[test]
    fn returns_empty_for_body_without_open_marker() {
        let parsed =
            parse("plain text", &glm47_markers(), &glm47_shape()).expect("must parse empty");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_when_shape_is_incomplete() {
        let mut shape = glm47_shape();
        shape.value_close.clear();
        let body =
            "<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>";
        let parsed = parse(body, &glm47_markers(), &shape).expect("must parse empty");
        assert!(parsed.is_empty());
    }
}
