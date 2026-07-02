use llama_cpp_bindings_types::bracketed_json_shape::BracketedJsonShape;
use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

pub struct Mistral3ArrowArgsOverride;

impl Mistral3ArrowArgsOverride {
    const TEMPLATE_FINGERPRINT: &'static str = "'[ARGS]'";

    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "[TOOL_CALLS]".to_owned(),
            close: String::new(),
            args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
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
    use llama_cpp_bindings_types::bracketed_json_shape::BracketedJsonShape;
    use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;

    use super::Mistral3ArrowArgsOverride;

    #[test]
    fn detects_mistral3_template_with_args_literal() {
        let template = "...{{- name + '[ARGS]' + arguments }}...";
        let markers = Mistral3ArrowArgsOverride::detect(template)
            .expect("Mistral 3 template must be detected");

        assert_eq!(markers.open, "[TOOL_CALLS]");
        assert!(markers.close.is_empty());
        assert_eq!(
            markers.args_shape,
            ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
            })
        );
    }

    #[test]
    fn returns_none_for_template_without_fingerprint() {
        assert!(Mistral3ArrowArgsOverride::detect("just some plain template body").is_none());
    }

    #[test]
    fn returns_none_for_empty_template() {
        assert!(Mistral3ArrowArgsOverride::detect("").is_none());
    }

    #[test]
    fn returns_none_when_fingerprint_substring_appears_without_jinja_apostrophes() {
        let template = "doc text mentioning the [ARGS] tag without quoting it as a literal";
        assert!(Mistral3ArrowArgsOverride::detect(template).is_none());
    }
}
