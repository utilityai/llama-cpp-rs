use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_bindings_types::XmlTagsShape;
use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_until;

use crate::error::XmlFunctionTagsFailure;

const fn shape_is_complete(shape: &XmlTagsShape) -> bool {
    !shape.function_open_prefix.is_empty()
        && !shape.function_close.is_empty()
        && !shape.parameter_open_prefix.is_empty()
        && !shape.parameter_close.is_empty()
}

fn trim_surrounding_newlines(input: &str) -> &str {
    input.trim_start_matches('\n').trim_end_matches('\n')
}

fn parameter_value_to_json(raw: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .unwrap_or_else(|| serde_json::Value::String(raw.to_owned()))
}

fn locate_tag_name_end(after_prefix: &str) -> Option<usize> {
    let close_position = after_prefix.find('>');
    let next_open_position = after_prefix.find('<');

    match (close_position, next_open_position) {
        (Some(close), Some(open)) if open < close => None,
        (Some(close), _) => Some(close),
        (None, _) => None,
    }
}

fn skip_to_next_function_open<'body>(
    input: &'body str,
    function_open_prefix: &str,
) -> Option<&'body str> {
    let take_result: IResult<&'body str, &'body str> =
        take_until(function_open_prefix).parse(input);
    let (after_prefix_inclusive, _) = take_result.ok()?;
    let consume_result: IResult<&'body str, &'body str> =
        tag(function_open_prefix).parse(after_prefix_inclusive);
    let (after_prefix, _) = consume_result.ok()?;

    Some(after_prefix)
}

fn parse_one_parameter<'body>(
    input: &'body str,
    shape: &XmlTagsShape,
    function_name: &str,
) -> Result<Option<(String, serde_json::Value, &'body str)>, XmlFunctionTagsFailure> {
    let take_result: IResult<&'body str, &'body str> =
        take_until(shape.parameter_open_prefix.as_str()).parse(input);
    let Ok((after_prefix_inclusive, _)) = take_result else {
        return Ok(None);
    };
    let consume_result: IResult<&'body str, &'body str> =
        tag(shape.parameter_open_prefix.as_str()).parse(after_prefix_inclusive);
    let Ok((after_prefix, _)) = consume_result else {
        return Ok(None);
    };

    let Some(name_end) = locate_tag_name_end(after_prefix) else {
        return Err(XmlFunctionTagsFailure::UnclosedParameterBlock {
            function_name: function_name.to_owned(),
            parameter_name: String::new(),
            expected_close: shape.parameter_close.clone(),
        });
    };
    let parameter_name = after_prefix[..name_end].trim().to_owned();
    if parameter_name.is_empty() {
        return Err(XmlFunctionTagsFailure::EmptyParameterName {
            function_name: function_name.to_owned(),
        });
    }
    let value_start = &after_prefix[name_end + 1..];

    let Some(value_end_position) = value_start.find(shape.parameter_close.as_str()) else {
        return Err(XmlFunctionTagsFailure::UnclosedParameterBlock {
            function_name: function_name.to_owned(),
            parameter_name,
            expected_close: shape.parameter_close.clone(),
        });
    };
    let raw_value = trim_surrounding_newlines(&value_start[..value_end_position]);
    let after_close = &value_start[value_end_position + shape.parameter_close.len()..];
    let parameter_value = parameter_value_to_json(raw_value);

    Ok(Some((parameter_name, parameter_value, after_close)))
}

fn collect_parameters(
    function_body: &str,
    shape: &XmlTagsShape,
    function_name: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, XmlFunctionTagsFailure> {
    let mut parameters = serde_json::Map::new();
    let mut remaining = function_body;

    while let Some((parameter_name, parameter_value, rest)) =
        parse_one_parameter(remaining, shape, function_name)?
    {
        parameters.insert(parameter_name, parameter_value);
        remaining = rest;
    }

    Ok(parameters)
}

fn parse_one_function<'body>(
    input: &'body str,
    shape: &XmlTagsShape,
) -> Result<Option<(ParsedToolCall, &'body str)>, XmlFunctionTagsFailure> {
    let Some(after_function_prefix) =
        skip_to_next_function_open(input, &shape.function_open_prefix)
    else {
        return Ok(None);
    };

    let Some(name_end) = locate_tag_name_end(after_function_prefix) else {
        return Err(XmlFunctionTagsFailure::UnclosedFunctionBlock {
            function_name: String::new(),
            expected_close: shape.function_close.clone(),
        });
    };
    let function_name = after_function_prefix[..name_end].trim().to_owned();
    if function_name.is_empty() {
        return Err(XmlFunctionTagsFailure::EmptyFunctionName);
    }
    let function_body_start = &after_function_prefix[name_end + 1..];

    let Some(function_body_end) = function_body_start.find(shape.function_close.as_str()) else {
        return Err(XmlFunctionTagsFailure::UnclosedFunctionBlock {
            function_name,
            expected_close: shape.function_close.clone(),
        });
    };
    let function_body = &function_body_start[..function_body_end];
    let after_function_close =
        &function_body_start[function_body_end + shape.function_close.len()..];

    let arguments_object = collect_parameters(function_body, shape, &function_name)?;
    let arguments_value = serde_json::Value::Object(arguments_object);
    let arguments = ToolCallArguments::from_string(arguments_value.to_string());

    Ok(Some((
        ParsedToolCall::new(String::new(), function_name, arguments),
        after_function_close,
    )))
}

/// # Errors
///
/// Returns [`XmlFunctionTagsFailure`] when the body looks like an XML
/// function-tag tool-call block (matches the function open prefix) but
/// contains a structural issue: empty function/parameter name or an
/// unclosed function/parameter block.
pub fn parse(
    body: &str,
    shape: &XmlTagsShape,
) -> Result<Vec<ParsedToolCall>, XmlFunctionTagsFailure> {
    if !shape_is_complete(shape) {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    let mut remaining = body;

    while let Some((call, rest)) = parse_one_function(remaining, shape)? {
        parsed.push(call);
        remaining = rest;
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArguments;
    use llama_cpp_bindings_types::XmlTagsShape;
    use serde_json::json;

    use super::parse;
    use crate::error::XmlFunctionTagsFailure;

    fn xml_shape() -> XmlTagsShape {
        XmlTagsShape {
            function_open_prefix: "<function=".to_owned(),
            function_close: "</function>".to_owned(),
            parameter_open_prefix: "<parameter=".to_owned(),
            parameter_close: "</parameter>".to_owned(),
        }
    }

    #[test]
    fn parses_single_function_with_one_parameter() {
        let body =
            "\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n";
        let parsed = parse(body, &xml_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn parses_function_with_multiple_parameters() {
        let body = "<function=f><parameter=a>1</parameter><parameter=b>two</parameter></function>";
        let parsed = parse(body, &xml_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"a": 1, "b": "two"})),
        );
    }

    #[test]
    fn parses_two_function_blocks_in_one_body() {
        let body = "<function=a><parameter=x>1</parameter></function><function=b><parameter=y>2</parameter></function>";
        let parsed = parse(body, &xml_shape()).expect("must parse");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert_eq!(parsed[1].name, "b");
    }

    #[test]
    fn preserves_multi_line_parameter_value() {
        let body = "<function=f>\n<parameter=msg>\nline one\nline two\n</parameter>\n</function>";
        let parsed = parse(body, &xml_shape()).expect("must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"msg": "line one\nline two"})),
        );
    }

    #[test]
    fn rejects_function_tag_missing_closing_angle_with_typed_failure() {
        let body = "<function=get_weather\n<parameter=location>Paris</parameter></function>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::UnclosedFunctionBlock { .. } => {}
            other => panic!("expected UnclosedFunctionBlock, got {other:?}"),
        }
    }

    #[test]
    fn rejects_function_block_missing_close_tag_with_typed_failure() {
        let body = "<function=get_weather><parameter=location>Paris</parameter>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::UnclosedFunctionBlock {
                function_name,
                expected_close,
            } => {
                assert_eq!(function_name, "get_weather");
                assert_eq!(expected_close, "</function>");
            }
            other => panic!("expected UnclosedFunctionBlock, got {other:?}"),
        }
    }

    #[test]
    fn rejects_parameter_tag_missing_closing_angle_with_typed_failure() {
        let body = "<function=f><parameter=x</function>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::UnclosedParameterBlock {
                function_name,
                parameter_name,
                expected_close,
            } => {
                assert_eq!(function_name, "f");
                assert_eq!(parameter_name, "");
                assert_eq!(expected_close, "</parameter>");
            }
            other => panic!("expected UnclosedParameterBlock, got {other:?}"),
        }
    }

    #[test]
    fn rejects_parameter_block_missing_close_tag_with_typed_failure() {
        let body = "<function=get_weather><parameter=location>Paris</function>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::UnclosedParameterBlock {
                function_name,
                parameter_name,
                expected_close,
            } => {
                assert_eq!(function_name, "get_weather");
                assert_eq!(parameter_name, "location");
                assert_eq!(expected_close, "</parameter>");
            }
            other => panic!("expected UnclosedParameterBlock, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_function_name_with_typed_failure() {
        let body = "<function=><parameter=x>1</parameter></function>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::EmptyFunctionName => {}
            other => panic!("expected EmptyFunctionName, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_parameter_name_with_typed_failure() {
        let body = "<function=f><parameter=>1</parameter></function>";
        let result = parse(body, &xml_shape());

        match result.expect_err("must error") {
            XmlFunctionTagsFailure::EmptyParameterName { function_name } => {
                assert_eq!(function_name, "f");
            }
            other => panic!("expected EmptyParameterName, got {other:?}"),
        }
    }

    #[test]
    fn returns_empty_when_body_has_no_function_tag() {
        let parsed =
            parse("plain text without function tags", &xml_shape()).expect("must parse empty");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_for_empty_body() {
        let parsed = parse("", &xml_shape()).expect("must parse empty");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_when_shape_has_empty_required_field() {
        let mut shape = xml_shape();
        shape.function_close.clear();
        let body = "<function=f><parameter=x>1</parameter></function>";
        let parsed = parse(body, &shape).expect("must parse empty");
        assert!(parsed.is_empty());
    }

    #[test]
    fn parses_negotiate_with_cat_reproducer_payload() {
        let body = "<tool_call>\n\
<function=negotiate_with_cat>\n\
<parameter=bribe>\n\
tuna\n\
</parameter>\n\
<parameter=desperation_level>\n\
8\n\
</parameter>\n\
<parameter=topic>\n\
get off the keyboard\n\
</parameter>\n\
</function>\n\
</tool_call>";
        let parsed = parse(body, &xml_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "negotiate_with_cat");
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({
                "bribe": "tuna",
                "desperation_level": 8,
                "topic": "get off the keyboard",
            })),
        );
    }
}
