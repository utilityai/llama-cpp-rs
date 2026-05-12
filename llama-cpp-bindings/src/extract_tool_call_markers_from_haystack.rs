use crate::tool_call_marker_pair::ToolCallMarkerPair;

#[must_use]
pub fn extract_tool_call_markers_from_haystack(haystack: &str) -> Option<ToolCallMarkerPair> {
    if haystack.is_empty() {
        return None;
    }

    let json_start = haystack.find('{')?;
    let json_end = haystack.rfind('}')?;
    if json_end < json_start {
        return None;
    }

    let json_slice = &haystack[json_start..=json_end];
    serde_json::from_str::<serde_json::Value>(json_slice).ok()?;

    let open = haystack[..json_start].trim().to_owned();
    let close = haystack[json_end + 1..].trim().to_owned();

    if open.is_empty() || close.is_empty() {
        return None;
    }

    Some(ToolCallMarkerPair { open, close })
}

#[cfg(test)]
mod tests {
    use super::ToolCallMarkerPair;
    use super::extract_tool_call_markers_from_haystack;

    #[test]
    fn extracts_open_and_close_around_a_simple_json_payload() {
        let pair = extract_tool_call_markers_from_haystack(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>",
        );

        assert_eq!(
            pair,
            Some(ToolCallMarkerPair {
                open: "<tool_call>".to_owned(),
                close: "</tool_call>".to_owned(),
            }),
        );
    }

    #[test]
    fn trims_surrounding_whitespace_from_each_marker() {
        let pair = extract_tool_call_markers_from_haystack(
            "  <tool_call>\n  {\"k\": 1}\n  </tool_call>  ",
        );

        assert_eq!(
            pair,
            Some(ToolCallMarkerPair {
                open: "<tool_call>".to_owned(),
                close: "</tool_call>".to_owned(),
            }),
        );
    }

    #[test]
    fn returns_none_when_haystack_is_empty() {
        assert_eq!(extract_tool_call_markers_from_haystack(""), None);
    }

    #[test]
    fn returns_none_when_haystack_has_no_open_brace() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("plain assistant text"),
            None
        );
    }

    #[test]
    fn returns_none_when_haystack_has_open_brace_but_no_close() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("<open>{ unclosed"),
            None
        );
    }

    #[test]
    fn returns_none_when_close_brace_precedes_open_brace() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("</close>}{<open>"),
            None
        );
    }

    #[test]
    fn returns_none_when_brace_payload_is_not_valid_json() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("<open>{not valid json}</close>"),
            None
        );
    }

    #[test]
    fn returns_none_when_open_marker_resolves_to_empty_after_trim() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("   {\"x\":1}</close>"),
            None
        );
    }

    #[test]
    fn returns_none_when_close_marker_resolves_to_empty_after_trim() {
        assert_eq!(
            extract_tool_call_markers_from_haystack("<open>{\"x\":1}   "),
            None
        );
    }

    #[test]
    fn extracts_around_an_object_that_contains_nested_braces() {
        let pair = extract_tool_call_markers_from_haystack(
            "<tool>{\"args\":{\"k\":[1,2,{\"deep\":true}]}}</tool>",
        );

        assert_eq!(
            pair,
            Some(ToolCallMarkerPair {
                open: "<tool>".to_owned(),
                close: "</tool>".to_owned(),
            }),
        );
    }

    #[test]
    fn extracts_when_open_marker_contains_multibyte_utf8() {
        let pair = extract_tool_call_markers_from_haystack("<|tool→call|>{\"k\":1}<|/tool→call|>");

        assert_eq!(
            pair,
            Some(ToolCallMarkerPair {
                open: "<|tool→call|>".to_owned(),
                close: "<|/tool→call|>".to_owned(),
            }),
        );
    }
}
