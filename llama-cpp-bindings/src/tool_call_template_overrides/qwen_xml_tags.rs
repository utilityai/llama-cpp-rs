use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;
use llama_cpp_bindings_types::XmlTagsShape;

const TEMPLATE_FINGERPRINT: &str = "<function=";

#[must_use]
pub fn markers() -> ToolCallMarkers {
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

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    if !template.contains(TEMPLATE_FINGERPRINT) {
        return None;
    }
    Some(markers())
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn detects_qwen_xml_template_with_function_tag_literal() {
        let template = "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}";
        let markers = detect(template).expect("Qwen XML template must be detected");

        assert_eq!(markers.open, "<tool_call>");
        assert_eq!(markers.close, "</tool_call>");
        let ToolCallArgsShape::XmlTags(shape) = markers.args_shape else {
            panic!("expected XmlTags variant, got {:?}", markers.args_shape);
        };
        assert_eq!(shape.function_open_prefix, "<function=");
        assert_eq!(shape.function_close, "</function>");
        assert_eq!(shape.parameter_open_prefix, "<parameter=");
        assert_eq!(shape.parameter_close, "</parameter>");
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
    fn detects_qwen_xml_template_with_concatenated_string_literal() {
        let template = "{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}";
        assert!(detect(template).is_some());
    }
}
