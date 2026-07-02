use llama_cpp_bindings_types::paired_quote_shape::PairedQuoteShape;
use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;
use llama_cpp_bindings_types::tool_call_value_quote::ToolCallValueQuote;

pub struct Gemma4CallBlockOverride;

impl Gemma4CallBlockOverride {
    const TEMPLATE_FINGERPRINT: &'static str = "'<|tool_call>call:'";

    #[must_use]
    pub fn markers() -> ToolCallMarkers {
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

    #[must_use]
    pub fn detect(template: &str) -> Option<ToolCallMarkers> {
        if !template.contains(Self::TEMPLATE_FINGERPRINT) {
            return None;
        }
        Some(Self::markers())
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;

    use super::Gemma4CallBlockOverride;

    #[test]
    fn detects_gemma4_template_with_tool_call_call_literal() {
        use llama_cpp_bindings_types::paired_quote_shape::PairedQuoteShape;
        use llama_cpp_bindings_types::tool_call_value_quote::ToolCallValueQuote;

        let template = "...{{- '<|tool_call>call:' + function['name'] + '{' -}}...";
        let markers =
            Gemma4CallBlockOverride::detect(template).expect("Gemma 4 template must be detected");

        assert_eq!(markers.open, "<|tool_call>call:");
        assert_eq!(markers.close, "}");
        assert_eq!(
            markers.args_shape,
            ToolCallArgsShape::PairedQuote(PairedQuoteShape {
                name_args_separator: "{".to_owned(),
                value_quote: ToolCallValueQuote {
                    open: "<|\"|>".to_owned(),
                    close: "<|\"|>".to_owned(),
                },
            })
        );
    }

    #[test]
    fn returns_none_for_template_without_fingerprint() {
        assert!(Gemma4CallBlockOverride::detect("just some plain template body").is_none());
    }

    #[test]
    fn returns_none_for_empty_template() {
        assert!(Gemma4CallBlockOverride::detect("").is_none());
    }

    #[test]
    fn returns_none_when_fingerprint_substring_appears_without_jinja_apostrophes() {
        let template = "doc explaining the <|tool_call>call: format in prose, not as a literal";
        assert!(Gemma4CallBlockOverride::detect(template).is_none());
    }
}
