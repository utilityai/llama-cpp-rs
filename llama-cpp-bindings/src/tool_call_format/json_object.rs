use llama_cpp_bindings_types::JsonObjectShape;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ToolCallArguments;

use crate::error::JsonObjectFailure;

fn try_parse_one_object(
    input: &str,
    shape: &JsonObjectShape,
) -> Result<Option<(ParsedToolCall, usize)>, JsonObjectFailure> {
    let trimmed_start = input.find('{');
    let Some(start) = trimmed_start else {
        return Ok(None);
    };

    let mut stream = serde_json::Deserializer::from_str(&input[start..])
        .into_iter::<serde_json::Map<String, serde_json::Value>>();
    let map = match stream.next() {
        Some(Ok(map)) => map,
        Some(Err(err)) => {
            return Err(JsonObjectFailure::InvalidJson {
                message: err.to_string(),
            });
        }
        None => return Ok(None),
    };
    let consumed = stream.byte_offset();

    let Some(name_value) = map.get(&shape.name_field) else {
        return Ok(None);
    };
    let serde_json::Value::String(name) = name_value else {
        return Ok(None);
    };

    let arguments_value = map
        .get(&shape.arguments_field)
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    let arguments = ToolCallArguments::from_string(arguments_value.to_string());

    let trailing_extras = map
        .keys()
        .any(|key| key != &shape.name_field && key != &shape.arguments_field);
    if trailing_extras {
        return Ok(None);
    }

    Ok(Some((
        ParsedToolCall::new(String::new(), name.clone(), arguments),
        start + consumed,
    )))
}

/// # Errors
///
/// Returns [`JsonObjectFailure`] when the body contains a JSON object that
/// looks like a tool call (matches the open brace at start) but the JSON itself
/// is malformed.
pub fn parse(
    body: &str,
    shape: &JsonObjectShape,
) -> Result<Vec<ParsedToolCall>, JsonObjectFailure> {
    if shape.name_field.is_empty() || shape.arguments_field.is_empty() {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    let mut remaining = body;

    while let Some((call, consumed)) = try_parse_one_object(remaining, shape)? {
        parsed.push(call);
        remaining = &remaining[consumed..];
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::JsonObjectShape;
    use llama_cpp_bindings_types::ToolCallArguments;
    use serde_json::json;

    use super::parse;
    use crate::error::JsonObjectFailure;

    fn qwen3_shape() -> JsonObjectShape {
        JsonObjectShape {
            name_field: "name".to_owned(),
            arguments_field: "arguments".to_owned(),
        }
    }

    #[test]
    fn parses_single_json_object_with_name_and_arguments() {
        let parsed = parse(
            r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#,
            &qwen3_shape(),
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
    fn parses_json_object_after_leading_whitespace_and_newlines() {
        let parsed = parse(
            "\n  {\"name\": \"f\", \"arguments\": {\"a\": 1}}\n",
            &qwen3_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "f");
    }

    #[test]
    fn parses_two_consecutive_json_objects() {
        let parsed = parse(
            r#"{"name": "a", "arguments": {}}{"name": "b", "arguments": {"x": 2}}"#,
            &qwen3_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert_eq!(parsed[1].name, "b");
    }

    #[test]
    fn parses_object_with_arguments_field_missing_yields_empty_arguments() {
        let parsed = parse(r#"{"name": "ping"}"#, &qwen3_shape()).expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "ping");
        assert_eq!(parsed[0].arguments, ToolCallArguments::ValidJson(json!({})),);
    }

    #[test]
    fn rejects_json_object_with_extra_unexpected_top_level_keys() {
        let parsed = parse(
            r#"{"name": "f", "arguments": {}, "extra": 1}"#,
            &qwen3_shape(),
        )
        .expect("must parse");

        assert!(parsed.is_empty(), "extra top-level key must reject");
    }

    #[test]
    fn rejects_json_object_with_non_string_name() {
        let parsed =
            parse(r#"{"name": 123, "arguments": {}}"#, &qwen3_shape()).expect("must parse");

        assert!(parsed.is_empty(), "non-string name must reject");
    }

    #[test]
    fn rejects_input_without_open_brace() {
        let parsed = parse("plain content", &qwen3_shape()).expect("must parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn rejects_array_instead_of_object() {
        let parsed = parse("[1, 2, 3]", &qwen3_shape()).expect("must parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_failure_for_malformed_json() {
        let err = parse(r#"{"name": "f", "arguments": {"a": }"#, &qwen3_shape()).unwrap_err();
        let JsonObjectFailure::InvalidJson { message } = err;

        assert!(!message.is_empty());
    }

    #[test]
    fn returns_empty_when_object_is_not_a_tool_call_shape() {
        let parsed = parse("{ \"foo\": 1 }", &qwen3_shape()).expect("must parse");

        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_when_shape_has_empty_required_field() {
        let mut shape = qwen3_shape();
        shape.name_field.clear();
        let parsed = parse(r#"{"name": "x", "arguments": {}}"#, &shape).expect("must parse");
        assert!(parsed.is_empty());
    }
}
