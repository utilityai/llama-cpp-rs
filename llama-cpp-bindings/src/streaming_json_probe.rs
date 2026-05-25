use serde_json::Value;
use serde_json::error::Category;

const NAME_FIELD: &str = "name";
const ARGUMENTS_FIELD: &str = "arguments";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JsonProbeOutcome {
    StillPossiblyValid,
    CompletedValid,
    Failed,
}

impl JsonProbeOutcome {
    #[must_use]
    pub fn validate_prefix(buffer: &str) -> Self {
        let trimmed = buffer.trim_start();
        if trimmed.is_empty() {
            return Self::StillPossiblyValid;
        }
        if !trimmed.starts_with('{') {
            return Self::Failed;
        }

        let mut stream = serde_json::Deserializer::from_str(trimmed).into_iter::<Value>();
        match stream.next() {
            Some(Ok(value)) => evaluate_completed_value(&value, &trimmed[stream.byte_offset()..]),
            Some(Err(parse_error)) => match parse_error.classify() {
                Category::Eof => Self::StillPossiblyValid,
                Category::Io | Category::Syntax | Category::Data => Self::Failed,
            },
            None => Self::StillPossiblyValid,
        }
    }
}

fn evaluate_completed_value(value: &Value, trailing: &str) -> JsonProbeOutcome {
    let Value::Object(map) = value else {
        return JsonProbeOutcome::Failed;
    };

    let Some(Value::String(name)) = map.get(NAME_FIELD) else {
        return JsonProbeOutcome::Failed;
    };
    if name.is_empty() {
        return JsonProbeOutcome::Failed;
    }

    if let Some(arguments) = map.get(ARGUMENTS_FIELD)
        && !matches!(arguments, Value::Object(_))
    {
        return JsonProbeOutcome::Failed;
    }

    for key in map.keys() {
        if key != NAME_FIELD && key != ARGUMENTS_FIELD {
            return JsonProbeOutcome::Failed;
        }
    }

    if trailing.trim().is_empty() {
        JsonProbeOutcome::CompletedValid
    } else {
        JsonProbeOutcome::Failed
    }
}

#[cfg(test)]
mod tests {
    use super::JsonProbeOutcome;

    #[test]
    fn empty_buffer_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(""),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn whitespace_only_buffer_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("   \n  "),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn single_open_brace_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("{"),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn open_brace_with_trailing_space_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("{ "),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn open_brace_with_quote_starting_key_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ ""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn partial_name_key_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn partial_name_value_quote_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": ""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn partial_name_value_letters_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "ge"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn complete_name_string_no_comma_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "get_weather""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_comma_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "get_weather","#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_partial_arguments_key_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "get_weather", "argum"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_arguments_key_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "get_weather", "arguments""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_arguments_open_brace_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "get_weather", "arguments": {"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn arguments_with_partial_inner_key_value_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(
                r#"{ "name": "get_weather", "arguments": {"location":"#
            ),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn arguments_with_partial_inner_string_value_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(
                r#"{ "name": "get_weather", "arguments": {"location": "Pa"#
            ),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn complete_simple_tool_call_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_internal_whitespace_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name": "f", "arguments": {}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_string_argument_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(
                r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#
            ),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_multiple_arguments_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(
                r#"{"name":"book_flight","arguments":{"from":"NYC","to":"PAR","passengers":2}}"#
            ),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_nested_arguments_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{"a":{"b":[1,2,3]}}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_close_brace_inside_string_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{"q":"a } b"}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_escaped_quotes_in_string_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{"q":"he said \"hi\""}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_unicode_strings_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"日本語","arguments":{"city":"パリ"}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_trailing_whitespace_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("{\"name\":\"f\",\"arguments\":{}}\n"),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_array_inside_arguments_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{"items":[1,2,3]}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_without_arguments_field_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"ping"}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn top_level_array_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("["),
            JsonProbeOutcome::Failed
        );
    }

    #[test]
    fn top_level_scalar_number_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("123"),
            JsonProbeOutcome::Failed
        );
    }

    #[test]
    fn top_level_string_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#""hi""#),
            JsonProbeOutcome::Failed
        );
    }

    #[test]
    fn complete_object_with_wrong_first_key_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"foo":"bar"}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_non_string_name_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":123,"arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_null_name_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":null,"arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_arguments_as_array_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":[]}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_arguments_as_string_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":"hi"}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_third_top_level_key_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{},"extra":1}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_empty_name_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"","arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_trailing_garbage_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":{}}garbage"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn empty_object_is_failed_due_to_missing_required_name() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("{}"),
            JsonProbeOutcome::Failed
        );
    }

    #[test]
    fn complete_object_with_arguments_only_no_name_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn leading_whitespace_then_open_brace_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("\n  \n{"),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn leading_whitespace_then_complete_tool_call_is_completed_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("\n  {\"name\":\"f\",\"arguments\":{}}"),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_followed_by_second_object_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(
                r#"{"name":"a","arguments":{}}{"name":"b","arguments":{}}"#
            ),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn buffer_with_only_open_quote_is_still_possibly_valid() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "n"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn buffer_with_complete_first_field_unknown_second_key_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{ "name": "f", "foo": 1}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn unicode_letter_inside_name_value_completes_validly() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"éclair","arguments":{}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn arguments_field_with_explicit_null_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix(r#"{"name":"f","arguments":null}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn syntactically_malformed_object_is_failed() {
        assert_eq!(
            JsonProbeOutcome::validate_prefix("{,}"),
            JsonProbeOutcome::Failed,
        );
    }
}
