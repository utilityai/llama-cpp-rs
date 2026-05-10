use llama_cpp_bindings_types::JsonObjectShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

const TEMPLATE_FINGERPRINT_OPEN: &str = "'<tool_call>\\n{\"name\": \"'";
const TEMPLATE_FINGERPRINT_ARGS_JOIN: &str = "'\", \"arguments\": '";

#[must_use]
pub fn markers() -> ToolCallMarkers {
    ToolCallMarkers {
        open: "<tool_call>".to_owned(),
        close: "</tool_call>".to_owned(),
        args_shape: ToolCallArgsShape::JsonObject(JsonObjectShape {
            name_field: "name".to_owned(),
            arguments_field: "arguments".to_owned(),
        }),
    }
}

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    if !template.contains(TEMPLATE_FINGERPRINT_OPEN) {
        return None;
    }
    if !template.contains(TEMPLATE_FINGERPRINT_ARGS_JOIN) {
        return None;
    }
    Some(markers())
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn detects_qwen3_json_inside_tool_call_template() {
        let template = "{{- '<tool_call>\\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' + (tool_call.arguments | tojson) + '}\\n</tool_call>' -}}";
        let markers = detect(template).expect("Qwen 3 template must be detected");

        assert_eq!(markers.open, "<tool_call>");
        assert_eq!(markers.close, "</tool_call>");
        let ToolCallArgsShape::JsonObject(shape) = markers.args_shape else {
            panic!("expected JsonObject variant, got {:?}", markers.args_shape);
        };
        assert_eq!(shape.name_field, "name");
        assert_eq!(shape.arguments_field, "arguments");
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
    fn returns_none_when_only_open_fingerprint_present() {
        let template = "{{- '<tool_call>\\n{\"name\": \"' + tool_call.name + ...";
        assert!(
            detect(template).is_none(),
            "open fingerprint alone must not match (Qwen3-Embedding-style false positive)",
        );
    }

    #[test]
    fn returns_none_when_only_args_join_fingerprint_present() {
        let template = "some text '\", \"arguments\": ' more text";
        assert!(detect(template).is_none());
    }
}
