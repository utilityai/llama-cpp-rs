pub mod bracketed_args;
pub mod json_object;
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
        ToolCallArgsShape::JsonObject(shape) => json_object::parse(body, shape).map_err(Into::into),
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

    #[test]
    fn try_parse_returns_no_match_for_glm_input_under_qwen_markers() {
        let glm_input = "<tool_call>get_weather\
            <arg_key>location</arg_key>\
            <arg_value>Paris</arg_value>\
            </tool_call>";

        match try_parse(glm_input, &qwen35_markers()) {
            ToolCallFormatOutcome::NoMatch => {}
            other => panic!("expected NoMatch for GLM input under Qwen markers, got {other:?}"),
        }
    }

    #[test]
    fn try_parse_returns_no_match_for_plain_content_under_every_known_shape() {
        use crate::tool_call_template_overrides::known_marker_candidates;

        let plain_content = "Sorry, I cannot help with that request.";

        for candidate in known_marker_candidates() {
            match try_parse(plain_content, &candidate) {
                ToolCallFormatOutcome::NoMatch => {}
                other => panic!(
                    "expected NoMatch for plain content under candidate {candidate:?}, got {other:?}"
                ),
            }
        }
    }

    #[test]
    fn duck_type_resolves_qwen_xml_input_via_xml_tags_shape_first() {
        use llama_cpp_bindings_types::ToolCallArguments;

        use crate::tool_call_template_overrides::known_marker_candidates;

        let qwen_input = "<tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>\n\
            Paris\n\
            </parameter>\n\
            </function>\n\
            </tool_call>";

        let mut resolved = None;
        for candidate in known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(calls) = try_parse(qwen_input, &candidate) {
                resolved = Some((candidate.args_shape, calls));
                break;
            }
        }

        let (args_shape, calls) =
            resolved.expect("Qwen XML input must resolve via at least one duck-type candidate");
        assert!(
            matches!(args_shape, ToolCallArgsShape::XmlTags(_)),
            "duck-type ordering must resolve Qwen XML via the XmlTags shape (most restrictive \
             shape that requires `<function=`), got {args_shape:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn duck_type_resolves_glm_input_via_key_value_xml_tags_shape() {
        use llama_cpp_bindings_types::ToolCallArguments;

        use crate::tool_call_template_overrides::known_marker_candidates;

        let glm_input = "<tool_call>get_weather\
            <arg_key>location</arg_key>\
            <arg_value>Paris</arg_value>\
            </tool_call>";

        let mut resolved = None;
        for candidate in known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(calls) = try_parse(glm_input, &candidate) {
                resolved = Some((candidate.args_shape, calls));
                break;
            }
        }

        let (args_shape, calls) =
            resolved.expect("GLM input must resolve via at least one duck-type candidate");
        assert!(
            matches!(args_shape, ToolCallArgsShape::KeyValueXmlTags(_)),
            "GLM input must resolve via the KeyValueXmlTags shape, got {args_shape:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn duck_type_resolves_mistral_input_via_bracketed_json_shape() {
        use llama_cpp_bindings_types::ToolCallArguments;

        use crate::tool_call_template_overrides::known_marker_candidates;

        let mistral_input = r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#;

        let mut resolved = None;
        for candidate in known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(calls) = try_parse(mistral_input, &candidate) {
                resolved = Some((candidate.args_shape, calls));
                break;
            }
        }

        let (args_shape, calls) =
            resolved.expect("Mistral input must resolve via at least one duck-type candidate");
        assert!(
            matches!(args_shape, ToolCallArgsShape::BracketedJson(_)),
            "Mistral input must resolve via the BracketedJson shape; the candidate ordering must \
             try BracketedJson before PairedQuote because PairedQuote's `{{` separator could \
             greedily match Mistral's JSON args. Got {args_shape:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }

    #[test]
    fn duck_type_resolves_gemma_input_via_paired_quote_shape() {
        use llama_cpp_bindings_types::ToolCallArguments;

        use crate::tool_call_template_overrides::known_marker_candidates;

        let gemma_input = "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";

        let mut resolved = None;
        for candidate in known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(calls) = try_parse(gemma_input, &candidate) {
                resolved = Some((candidate.args_shape, calls));
                break;
            }
        }

        let (args_shape, calls) =
            resolved.expect("Gemma input must resolve via at least one duck-type candidate");
        assert!(
            matches!(args_shape, ToolCallArgsShape::PairedQuote(_)),
            "Gemma input must resolve via the PairedQuote shape, got {args_shape:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        );
    }
}
