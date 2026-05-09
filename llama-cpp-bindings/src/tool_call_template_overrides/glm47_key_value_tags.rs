use llama_cpp_bindings_types::KeyValueXmlTagsShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

const TEMPLATE_FINGERPRINT: &str = "<arg_key>";

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    if !template.contains(TEMPLATE_FINGERPRINT) {
        return None;
    }
    Some(ToolCallMarkers {
        open: "<tool_call>".to_owned(),
        close: "</tool_call>".to_owned(),
        args_shape: ToolCallArgsShape::KeyValueXmlTags(KeyValueXmlTagsShape {
            key_open: "<arg_key>".to_owned(),
            key_close: "</arg_key>".to_owned(),
            value_open: "<arg_value>".to_owned(),
            value_close: "</arg_value>".to_owned(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn detects_glm47_template_with_arg_key_literal() {
        let template = "{{- '<tool_call>' + tool_call.name }}{% for k, v in args.items() %}<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>{% endfor %}</tool_call>";
        let markers = detect(template).expect("GLM-4.7 template must be detected");

        assert_eq!(markers.open, "<tool_call>");
        assert_eq!(markers.close, "</tool_call>");
        let ToolCallArgsShape::KeyValueXmlTags(shape) = markers.args_shape else {
            panic!(
                "expected KeyValueXmlTags variant, got {:?}",
                markers.args_shape
            );
        };
        assert_eq!(shape.key_open, "<arg_key>");
        assert_eq!(shape.key_close, "</arg_key>");
        assert_eq!(shape.value_open, "<arg_value>");
        assert_eq!(shape.value_close, "</arg_value>");
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
