use llama_cpp_bindings_types::json_object_shape::JsonObjectShape;
use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

pub struct Qwen3JsonInsideToolCallOverride;

impl Qwen3JsonInsideToolCallOverride {
    const TEMPLATE_FINGERPRINT_OPEN: &'static str = "'<tool_call>\\n{\"name\": \"'";
    const TEMPLATE_FINGERPRINT_ARGS_JOIN: &'static str = "'\", \"arguments\": '";

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
        if !template.contains(Self::TEMPLATE_FINGERPRINT_OPEN) {
            return None;
        }
        if !template.contains(Self::TEMPLATE_FINGERPRINT_ARGS_JOIN) {
            return None;
        }
        Some(Self::markers())
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::json_object_shape::JsonObjectShape;
    use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;

    use super::Qwen3JsonInsideToolCallOverride;

    #[test]
    fn detects_qwen3_json_inside_tool_call_template() {
        let template = "{{- '<tool_call>\\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' + (tool_call.arguments | tojson) + '}\\n</tool_call>' -}}";
        let markers = Qwen3JsonInsideToolCallOverride::detect(template)
            .expect("Qwen 3 template must be detected");

        assert_eq!(markers.open, "<tool_call>");
        assert_eq!(markers.close, "</tool_call>");
        assert_eq!(
            markers.args_shape,
            ToolCallArgsShape::JsonObject(JsonObjectShape {
                name_field: "name".to_owned(),
                arguments_field: "arguments".to_owned(),
            })
        );
    }

    #[test]
    fn returns_none_for_template_without_fingerprint() {
        assert!(Qwen3JsonInsideToolCallOverride::detect("just some plain template body").is_none());
    }

    #[test]
    fn returns_none_for_empty_template() {
        assert!(Qwen3JsonInsideToolCallOverride::detect("").is_none());
    }

    #[test]
    fn returns_none_when_only_open_fingerprint_present() {
        let template = "{{- '<tool_call>\\n{\"name\": \"' + tool_call.name + ...";
        assert!(
            Qwen3JsonInsideToolCallOverride::detect(template).is_none(),
            "open fingerprint alone must not match (Qwen3-Embedding-style false positive)",
        );
    }

    #[test]
    fn returns_none_when_only_args_join_fingerprint_present() {
        let template = "some text '\", \"arguments\": ' more text";
        assert!(Qwen3JsonInsideToolCallOverride::detect(template).is_none());
    }
}
