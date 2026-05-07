pub mod gemma4_call_block;
pub mod mistral3_arrow_args;
pub mod qwen_xml_tags;

use llama_cpp_bindings_types::ToolCallMarkers;

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    let detectors: [fn(&str) -> Option<ToolCallMarkers>; 3] = [
        gemma4_call_block::detect,
        mistral3_arrow_args::detect,
        qwen_xml_tags::detect,
    ];
    detectors
        .into_iter()
        .find_map(|detector| detector(template))
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
}
