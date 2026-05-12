use llama_cpp_bindings_types::ToolCallMarkers;

use crate::tool_call_template_overrides::gemma4_call_block::Gemma4CallBlockOverride;
use crate::tool_call_template_overrides::glm47_key_value_tags::Glm47KeyValueTagsOverride;
use crate::tool_call_template_overrides::mistral3_arrow_args::Mistral3ArrowArgsOverride;
use crate::tool_call_template_overrides::qwen_xml_tags::QwenXmlTagsOverride;
use crate::tool_call_template_overrides::qwen3_json_inside_tool_call::Qwen3JsonInsideToolCallOverride;

#[must_use]
pub fn known_marker_candidates() -> Vec<ToolCallMarkers> {
    vec![
        Qwen3JsonInsideToolCallOverride::markers(),
        QwenXmlTagsOverride::markers(),
        Glm47KeyValueTagsOverride::markers(),
        Mistral3ArrowArgsOverride::markers(),
        Gemma4CallBlockOverride::markers(),
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::known_marker_candidates;

    #[test]
    fn known_marker_candidates_returns_one_per_registered_shape() {
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
