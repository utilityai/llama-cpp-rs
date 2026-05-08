pub mod bracketed_args;
pub mod key_value_xml_tags;
pub mod paired_quote_args;
pub mod tool_call_format_outcome;
pub mod xml_function_tags;

pub use self::tool_call_format_outcome::ToolCallFormatOutcome;

use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

use crate::error::ToolCallFormatFailure;

#[must_use]
pub fn try_parse(body: &str, markers: &ToolCallMarkers) -> ToolCallFormatOutcome {
    if markers.open.is_empty() {
        return ToolCallFormatOutcome::NoMatch;
    }

    let parsed: Result<Vec<_>, ToolCallFormatFailure> = match &markers.args_shape {
        ToolCallArgsShape::BracketedJson(shape) => {
            bracketed_args::parse(body, markers, shape).map_err(Into::into)
        }
        ToolCallArgsShape::KeyValueXmlTags(shape) => {
            key_value_xml_tags::parse(body, markers, shape).map_err(Into::into)
        }
        ToolCallArgsShape::PairedQuote(shape) => {
            paired_quote_args::parse(body, markers, shape).map_err(Into::into)
        }
        ToolCallArgsShape::XmlTags(shape) => {
            xml_function_tags::parse(body, shape).map_err(Into::into)
        }
    };

    match parsed {
        Ok(parsed) if parsed.is_empty() => ToolCallFormatOutcome::NoMatch,
        Ok(parsed) => ToolCallFormatOutcome::Parsed(parsed),
        Err(failure) => ToolCallFormatOutcome::Failed(failure),
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::BracketedJsonShape;
    use llama_cpp_bindings_types::KeyValueXmlTagsShape;
    use llama_cpp_bindings_types::PairedQuoteShape;
    use llama_cpp_bindings_types::ToolCallArgsShape;
    use llama_cpp_bindings_types::ToolCallArguments;
    use llama_cpp_bindings_types::ToolCallMarkers;
    use llama_cpp_bindings_types::ToolCallValueQuote;
    use llama_cpp_bindings_types::XmlTagsShape;
    use serde_json::json;

    use super::ToolCallFormatOutcome;
    use super::try_parse;

    fn mistral3_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "[TOOL_CALLS]".to_owned(),
            close: String::new(),
            args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
            }),
        }
    }

    fn gemma4_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<|tool_call>call:".to_owned(),
            close: "}".to_owned(),
            args_shape: ToolCallArgsShape::PairedQuote(PairedQuoteShape {
                name_args_separator: "{".to_owned(),
                value_quote: ToolCallValueQuote {
                    open: "<|\"|>".to_owned(),
                    close: "<|\"|>".to_owned(),
                },
            }),
        }
    }

    fn qwen35_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::XmlTags(XmlTagsShape {
                function_open_prefix: "<function=".to_owned(),
                function_close: "</function>".to_owned(),
                parameter_open_prefix: "<parameter=".to_owned(),
                parameter_close: "</parameter>".to_owned(),
            }),
        }
    }

    fn glm47_markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::KeyValueXmlTags(KeyValueXmlTagsShape {
                key_open: "<arg_key>".to_owned(),
                key_close: "</arg_key>".to_owned(),
                value_open: "<arg_value>".to_owned(),
                value_close: "</arg_value>".to_owned(),
            }),
        }
    }

    #[test]
    fn dispatches_to_bracketed_args_for_mistral3_shape() {
        let outcome = try_parse(
            "[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}",
            &mistral3_markers(),
        );

        match outcome {
            ToolCallFormatOutcome::Parsed(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(
                    calls[0].arguments,
                    ToolCallArguments::ValidJson(json!({"location": "Paris"})),
                );
            }
            other => panic!("expected Parsed, got {other:?}"),
        }
    }

    #[test]
    fn dispatches_to_paired_quote_args_for_gemma4_shape() {
        let outcome = try_parse(
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}",
            &gemma4_markers(),
        );

        match outcome {
            ToolCallFormatOutcome::Parsed(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(
                    calls[0].arguments,
                    ToolCallArguments::ValidJson(json!({"location": "Paris"})),
                );
            }
            other => panic!("expected Parsed, got {other:?}"),
        }
    }

    #[test]
    fn dispatches_to_key_value_xml_tags_for_glm47_shape() {
        let outcome = try_parse(
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
            &glm47_markers(),
        );

        match outcome {
            ToolCallFormatOutcome::Parsed(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(
                    calls[0].arguments,
                    ToolCallArguments::ValidJson(json!({"location": "Paris"})),
                );
            }
            other => panic!("expected Parsed, got {other:?}"),
        }
    }

    #[test]
    fn dispatches_to_xml_function_tags_for_qwen35_shape() {
        let outcome = try_parse(
            "<function=get_weather><parameter=location>Paris</parameter></function>",
            &qwen35_markers(),
        );

        match outcome {
            ToolCallFormatOutcome::Parsed(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(
                    calls[0].arguments,
                    ToolCallArguments::ValidJson(json!({"location": "Paris"})),
                );
            }
            other => panic!("expected Parsed, got {other:?}"),
        }
    }

    #[test]
    fn no_match_when_open_marker_is_empty() {
        let markers = ToolCallMarkers {
            open: String::new(),
            close: String::new(),
            args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
            }),
        };

        match try_parse("[TOOL_CALLS]get_weather[ARGS]{}", &markers) {
            ToolCallFormatOutcome::NoMatch => {}
            other => panic!("expected NoMatch, got {other:?}"),
        }
    }

    #[test]
    fn no_match_when_body_lacks_markers() {
        match try_parse("plain text without tool calls", &mistral3_markers()) {
            ToolCallFormatOutcome::NoMatch => {}
            other => panic!("expected NoMatch, got {other:?}"),
        }
    }

    #[test]
    fn failed_when_inner_parser_returns_typed_failure() {
        match try_parse(
            "[TOOL_CALLS]get_weather[ARGS]{\"location\":}",
            &mistral3_markers(),
        ) {
            ToolCallFormatOutcome::Failed(_) => {}
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}
