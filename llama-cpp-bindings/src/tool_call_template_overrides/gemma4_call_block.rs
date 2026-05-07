use llama_cpp_bindings_types::PairedQuoteShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;
use llama_cpp_bindings_types::ToolCallValueQuote;

const TEMPLATE_FINGERPRINT: &str = "'<|tool_call>call:'";

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    if !template.contains(TEMPLATE_FINGERPRINT) {
        return None;
    }
    Some(ToolCallMarkers {
        open: "<|tool_call>call:".to_owned(),
        close: "}".to_owned(),
        args_shape: ToolCallArgsShape::PairedQuote(PairedQuoteShape {
            name_args_separator: "{".to_owned(),
            value_quote: ToolCallValueQuote {
                open: "<|\"|>".to_owned(),
                close: "<|\"|>".to_owned(),
            },
        }),
    })
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn detects_gemma4_template_with_tool_call_call_literal() {
        let template = "...{{- '<|tool_call>call:' + function['name'] + '{' -}}...";
        let markers = detect(template).expect("Gemma 4 template must be detected");

        assert_eq!(markers.open, "<|tool_call>call:");
        assert_eq!(markers.close, "}");
        let ToolCallArgsShape::PairedQuote(shape) = markers.args_shape else {
            panic!("expected PairedQuote variant, got {:?}", markers.args_shape);
        };
        assert_eq!(shape.name_args_separator, "{");
        assert_eq!(shape.value_quote.open, "<|\"|>");
        assert_eq!(shape.value_quote.close, "<|\"|>");
    }

    #[test]
    fn returns_none_for_template_without_fingerprint() {
        assert!(detect("just some plain template body").is_none());
    }

    #[test]
    fn returns_none_for_empty_template() {
        assert!(detect("").is_none());
    }

    #[test]
    fn returns_none_when_fingerprint_substring_appears_without_jinja_apostrophes() {
        let template = "doc explaining the <|tool_call>call: format in prose, not as a literal";
        assert!(detect(template).is_none());
    }
}
