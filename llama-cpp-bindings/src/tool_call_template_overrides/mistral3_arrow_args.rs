use llama_cpp_bindings_types::BracketedJsonShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

const TEMPLATE_FINGERPRINT: &str = "'[ARGS]'";

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    if !template.contains(TEMPLATE_FINGERPRINT) {
        return None;
    }
    Some(ToolCallMarkers {
        open: "[TOOL_CALLS]".to_owned(),
        close: String::new(),
        args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
            name_args_separator: "[ARGS]".to_owned(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn detects_mistral3_template_with_args_literal() {
        let template = "...{{- name + '[ARGS]' + arguments }}...";
        let markers = detect(template).expect("Mistral 3 template must be detected");

        assert_eq!(markers.open, "[TOOL_CALLS]");
        assert!(markers.close.is_empty());
        let ToolCallArgsShape::BracketedJson(shape) = markers.args_shape else {
            panic!("expected BracketedJson variant, got {:?}", markers.args_shape);
        };
        assert_eq!(shape.name_args_separator, "[ARGS]");
    }

    #[test]
    fn returns_none_for_template_without_fingerprint() {
        assert!(detect("just some plain template body").is_none());
    }

    #[test]
    fn returns_none_for_empty_template() {
        assert!(detect("").is_none());
    }
}
