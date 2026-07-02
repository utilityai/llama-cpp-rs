use llama_cpp_bindings_types::paired_quote_shape::PairedQuoteShape;
use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;
use llama_cpp_bindings_types::tool_call_value_quote::ToolCallValueQuote;

use crate::error::paired_quote_failure::PairedQuoteFailure;

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

fn bare_value_to_json(text: &str) -> serde_json::Value {
    if text.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .unwrap_or_else(|| serde_json::Value::String(text.to_owned()))
}

fn find_bare_value_end(input: &str, close_marker: &str) -> usize {
    for (byte_index, character) in input.char_indices() {
        if character == ',' {
            return byte_index;
        }
        if !close_marker.is_empty() && input[byte_index..].starts_with(close_marker) {
            return byte_index;
        }
    }

    input.len()
}

fn parse_one_key<'body>(
    input: &'body str,
    tool_name: &str,
) -> Result<(String, &'body str), PairedQuoteFailure> {
    let Some((key_raw, after_colon)) = input.split_once(':') else {
        return Err(PairedQuoteFailure::UnclosedArgumentBlock {
            tool_name: tool_name.to_owned(),
            state: "key",
        });
    };
    let key = key_raw.trim().to_owned();
    if key.is_empty() {
        return Err(PairedQuoteFailure::EmptyKey {
            tool_name: tool_name.to_owned(),
        });
    }

    Ok((key, after_colon))
}

fn parse_one_value<'body>(
    input: &'body str,
    value_quote: &ToolCallValueQuote,
    close_marker: &str,
    tool_name: &str,
    key: &str,
) -> Result<(serde_json::Value, &'body str), PairedQuoteFailure> {
    let trimmed = input.trim_start();

    if !value_quote.open.is_empty()
        && !value_quote.close.is_empty()
        && let Some(after_open) = trimmed.strip_prefix(value_quote.open.as_str())
    {
        let Some(close_position) = after_open.find(value_quote.close.as_str()) else {
            return Err(PairedQuoteFailure::UnclosedQuotedValue {
                tool_name: tool_name.to_owned(),
                key: key.to_owned(),
            });
        };
        let value_text = after_open[..close_position].to_owned();
        let after_close = &after_open[close_position + value_quote.close.len()..];

        return Ok((serde_json::Value::String(value_text), after_close));
    }

    let bare_end = find_bare_value_end(trimmed, close_marker);
    let bare_text = trimmed[..bare_end].trim();
    let value = bare_value_to_json(bare_text);

    Ok((value, &trimmed[bare_end..]))
}

fn parse_args_body<'body>(
    input: &'body str,
    value_quote: &ToolCallValueQuote,
    close_marker: &str,
    tool_name: &str,
) -> Result<(serde_json::Map<String, serde_json::Value>, &'body str), PairedQuoteFailure> {
    let mut map = serde_json::Map::new();
    let mut remaining = input.trim_start();

    loop {
        if remaining.is_empty() {
            return Ok((map, remaining));
        }
        if !close_marker.is_empty()
            && let Some(after_close) = remaining.strip_prefix(close_marker)
        {
            return Ok((map, after_close));
        }

        let (key, after_key) = parse_one_key(remaining, tool_name)?;
        let (value, after_value) =
            parse_one_value(after_key, value_quote, close_marker, tool_name, &key)?;
        map.insert(key.clone(), value);

        remaining = after_value.trim_start();
        if !close_marker.is_empty()
            && let Some(after_close) = remaining.strip_prefix(close_marker)
        {
            return Ok((map, after_close));
        }
        if let Some(after_comma) = remaining.strip_prefix(',') {
            remaining = after_comma.trim_start();
            continue;
        }

        let Some(character) = remaining.chars().next() else {
            return Ok((map, remaining));
        };

        return Err(PairedQuoteFailure::UnexpectedCharAfterValue {
            tool_name: tool_name.to_owned(),
            key,
            character,
        });
    }
}

fn parse_one_call<'body>(
    input: &'body str,
    markers: &ToolCallMarkers,
    shape: &PairedQuoteShape,
) -> Result<ParseStep<'body>, PairedQuoteFailure> {
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

    let (args_object, after_args) = parse_args_body(
        after_separator,
        &shape.value_quote,
        markers.close.as_str(),
        &name,
    )?;
    let arguments_value = serde_json::Value::Object(args_object);

    Ok(ParseStep::Call(
        ParsedToolCall::new(
            String::new(),
            name,
            ToolCallArguments::ValidJson(arguments_value),
        ),
        after_args,
    ))
}

/// # Errors
///
/// Returns [`PairedQuoteFailure`] when the body looks like a paired-quote
/// tool-call block (matches the open marker and separator) but contains a
/// structural issue: empty key, unclosed quoted value, unexpected character
/// after a value, or an unfinished argument block.
pub fn parse(
    body: &str,
    markers: &ToolCallMarkers,
    shape: &PairedQuoteShape,
) -> Result<Vec<ParsedToolCall>, PairedQuoteFailure> {
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
    use llama_cpp_bindings_types::paired_quote_shape::PairedQuoteShape;
    use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
    use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;
    use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;
    use llama_cpp_bindings_types::tool_call_value_quote::ToolCallValueQuote;
    use serde_json::json;

    use super::parse;
    use crate::error::paired_quote_failure::PairedQuoteFailure;

    fn gemma4_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<|tool_call>call:".to_owned(),
            close: "}".to_owned(),
            args_shape: ToolCallArgsShape::PairedQuote(gemma4_shape()),
        }
    }

    fn gemma4_shape() -> PairedQuoteShape {
        PairedQuoteShape {
            name_args_separator: "{".to_owned(),
            value_quote: ToolCallValueQuote {
                open: "<|\"|>".to_owned(),
                close: "<|\"|>".to_owned(),
            },
        }
    }

    #[test]
    fn parses_single_quoted_string_argument_with_full_markers() {
        let parsed = parse(
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}",
            &gemma4_markers(),
            &gemma4_shape(),
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
    fn parses_classifier_stripped_body_without_open_or_close() {
        let parsed = parse(
            "get_weather{location:<|\"|>Paris<|\"|>",
            &gemma4_markers(),
            &gemma4_shape(),
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
    fn parses_multiple_quoted_string_arguments() {
        let parsed = parse(
            "<|tool_call>call:f{a:<|\"|>1<|\"|>,b:<|\"|>2<|\"|>}",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"a": "1", "b": "2"})),
        );
    }

    #[test]
    fn parses_bare_numeric_value() {
        let parsed = parse(
            "<|tool_call>call:f{a:42}",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"a": 42})),
        );
    }

    #[test]
    fn parses_bare_boolean_value() {
        let parsed = parse(
            "<|tool_call>call:f{a:true}",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"a": true})),
        );
    }

    #[test]
    fn rejects_unclosed_quoted_value_with_typed_failure() {
        let result = parse(
            "<|tool_call>call:f{a:<|\"|>oops",
            &gemma4_markers(),
            &gemma4_shape(),
        );

        assert_eq!(
            result.expect_err("unclosed quote must produce a typed failure"),
            PairedQuoteFailure::UnclosedQuotedValue {
                tool_name: "f".to_owned(),
                key: "a".to_owned(),
            },
        );
    }

    #[test]
    fn rejects_unexpected_char_after_value_with_typed_failure() {
        let result = parse(
            "<|tool_call>call:f{a:<|\"|>v<|\"|>$bad}",
            &gemma4_markers(),
            &gemma4_shape(),
        );

        assert_eq!(
            result.expect_err("garbage after value must produce a typed failure"),
            PairedQuoteFailure::UnexpectedCharAfterValue {
                tool_name: "f".to_owned(),
                key: "a".to_owned(),
                character: '$',
            },
        );
    }

    #[test]
    fn returns_empty_vec_for_empty_body() {
        let parsed = parse("", &gemma4_markers(), &gemma4_shape()).expect("empty body must parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_vec_when_body_lacks_separator() {
        let parsed = parse("no separator anywhere", &gemma4_markers(), &gemma4_shape())
            .expect("body without separator must parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn parses_args_body_terminated_by_end_of_input_after_quoted_value() {
        let parsed = parse(
            "<|tool_call>call:f{x:<|\"|>v<|\"|>",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("end-of-input after quoted value must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"x": "v"})),
        );
    }

    #[test]
    fn parses_args_body_terminated_by_end_of_input_after_bare_value() {
        let parsed = parse(
            "<|tool_call>call:f{n:42",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("end-of-input after bare value must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"n": 42})),
        );
    }

    #[test]
    fn rejects_empty_key_with_typed_failure() {
        let result = parse(
            "<|tool_call>call:f{:42}",
            &gemma4_markers(),
            &gemma4_shape(),
        );

        assert_eq!(
            result.expect_err("empty key must produce a typed failure"),
            PairedQuoteFailure::EmptyKey {
                tool_name: "f".to_owned(),
            },
        );
    }

    #[test]
    fn rejects_args_body_without_key_colon_with_typed_failure() {
        let result = parse(
            "<|tool_call>call:f{noColonHere",
            &gemma4_markers(),
            &gemma4_shape(),
        );

        assert_eq!(
            result.expect_err("args body without colon must produce a typed failure"),
            PairedQuoteFailure::UnclosedArgumentBlock {
                tool_name: "f".to_owned(),
                state: "key",
            },
        );
    }

    #[test]
    fn parses_empty_bare_value_as_null() {
        let parsed = parse("<|tool_call>call:f{a:}", &gemma4_markers(), &gemma4_shape())
            .expect("empty bare value must parse");

        assert_eq!(
            parsed[0].arguments,
            ToolCallArguments::ValidJson(json!({"a": null})),
        );
    }

    #[test]
    fn parses_call_with_empty_args_body_terminated_by_end_of_input() {
        let parsed = parse("<|tool_call>call:f{", &gemma4_markers(), &gemma4_shape())
            .expect("empty args body must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "f");
        assert_eq!(parsed[0].arguments, ToolCallArguments::ValidJson(json!({})),);
    }

    #[test]
    fn parses_call_with_empty_args_body_closed_by_marker() {
        let parsed = parse("<|tool_call>call:f{}", &gemma4_markers(), &gemma4_shape())
            .expect("empty args body closed by marker must parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "f");
        assert_eq!(parsed[0].arguments, ToolCallArguments::ValidJson(json!({})),);
    }

    #[test]
    fn stops_parsing_when_tool_name_is_empty() {
        let parsed = parse(
            "<|tool_call>call:{a:<|\"|>v<|\"|>}",
            &gemma4_markers(),
            &gemma4_shape(),
        )
        .expect("empty tool name must yield no calls");

        assert!(parsed.is_empty());
    }

    #[test]
    fn returns_empty_vec_when_separator_is_empty() {
        let shape = PairedQuoteShape {
            name_args_separator: String::new(),
            value_quote: ToolCallValueQuote {
                open: "<|\"|>".to_owned(),
                close: "<|\"|>".to_owned(),
            },
        };
        let parsed = parse(
            "<|tool_call>call:f{a:<|\"|>v<|\"|>}",
            &gemma4_markers(),
            &shape,
        )
        .expect("empty separator must yield no calls");

        assert!(parsed.is_empty());
    }
}
