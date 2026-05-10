pub mod gemma4_call_block;
pub mod glm47_key_value_tags;
pub mod mistral3_arrow_args;
pub mod qwen3_json_inside_tool_call;
pub mod qwen_xml_tags;

use llama_cpp_bindings_types::ToolCallMarkers;

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    let detectors: [fn(&str) -> Option<ToolCallMarkers>; 5] = [
        gemma4_call_block::detect,
        glm47_key_value_tags::detect,
        mistral3_arrow_args::detect,
        qwen3_json_inside_tool_call::detect,
        qwen_xml_tags::detect,
    ];
    detectors
        .into_iter()
        .find_map(|detector| detector(template))
}

#[must_use]
pub fn known_marker_candidates() -> Vec<ToolCallMarkers> {
    vec![
        qwen3_json_inside_tool_call::markers(),
        qwen_xml_tags::markers(),
        glm47_key_value_tags::markers(),
        mistral3_arrow_args::markers(),
        gemma4_call_block::markers(),
    ]
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::detect;

    #[test]
    fn dispatches_to_gemma4_override() {
        let template = "{{- '<|tool_call>call:' + function['name'] + '{' -}}";
        let markers = detect(template).expect("must dispatch to Gemma 4");

        assert_eq!(markers.open, "<|tool_call>call:");
        assert!(matches!(
            markers.args_shape,
            ToolCallArgsShape::PairedQuote(_)
        ));
    }

    #[test]
    fn dispatches_to_mistral3_override() {
        let template = "{{- name + '[ARGS]' + arguments }}";
        let markers = detect(template).expect("must dispatch to Mistral 3");

        assert_eq!(markers.open, "[TOOL_CALLS]");
        assert!(matches!(
            markers.args_shape,
            ToolCallArgsShape::BracketedJson(_)
        ));
    }

    #[test]
    fn dispatches_to_qwen_xml_tags_override() {
        let template = "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}";
        let markers = detect(template).expect("must dispatch to Qwen XML tags");

        assert_eq!(markers.open, "<tool_call>");
        assert!(matches!(markers.args_shape, ToolCallArgsShape::XmlTags(_)));
    }

    #[test]
    fn returns_none_when_no_override_matches() {
        assert!(detect("plain unrelated template").is_none());
    }

    #[test]
    fn known_marker_candidates_returns_one_per_registered_shape() {
        use std::collections::HashSet;

        use super::known_marker_candidates;

        let candidates = known_marker_candidates();
        assert_eq!(
            candidates.len(),
            5,
            "expected exactly five registered shapes, got {}",
            candidates.len()
        );

        let shape_discriminants: HashSet<&'static str> = candidates
            .iter()
            .map(|markers| match &markers.args_shape {
                ToolCallArgsShape::BracketedJson(_) => "BracketedJson",
                ToolCallArgsShape::JsonObject(_) => "JsonObject",
                ToolCallArgsShape::KeyValueXmlTags(_) => "KeyValueXmlTags",
                ToolCallArgsShape::PairedQuote(_) => "PairedQuote",
                ToolCallArgsShape::XmlTags(_) => "XmlTags",
            })
            .collect();
        assert_eq!(
            shape_discriminants.len(),
            5,
            "duplicate shape discriminants in known_marker_candidates: {shape_discriminants:?}"
        );
    }
}
