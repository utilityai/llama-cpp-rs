use serde_json::json;

use crate::ReasoningMarkers;

const REASON_PROBE: &str = "__PADDLER_REASON_PROBE_3F4A8C__";
const RESPONSE_PROBE: &str = "__PADDLER_RESPONSE_PROBE_3F4A8C__";

/// Baseline render messages, without a thinking chunk.
///
/// The assistant turn carries only the response sentinel; diffing the chunked
/// render against this baseline isolates the reasoning markers.
#[must_use]
pub fn plain_probe_messages_json() -> String {
    json!([
        { "role": "user", "content": "U" },
        { "role": "assistant", "content": RESPONSE_PROBE },
    ])
    .to_string()
}

/// Render messages whose assistant turn carries a thinking chunk.
///
/// The thinking chunk holds the reason sentinel and is followed by the response
/// sentinel, so diffing against the baseline surfaces the reasoning markers.
#[must_use]
pub fn chunked_probe_messages_json() -> String {
    json!([
        { "role": "user", "content": "U" },
        {
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": REASON_PROBE },
                { "type": "text", "text": RESPONSE_PROBE },
            ],
        },
    ])
    .to_string()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    find_subslice(haystack, needle).is_some()
}

/// Recovers the reasoning markers a chat template wraps around its thinking.
///
/// It diffs a render containing a thinking chunk against an otherwise identical
/// plain render (both produced by the C++ `llama_rs_render_chat_template`
/// primitive); this is the heuristic itself, isolated in Rust so it is
/// unit-testable on fixed render fixtures.
#[must_use]
pub fn extract_reasoning_markers_from_probe_renders(
    plain_render: &str,
    chunked_render: &str,
) -> Option<ReasoningMarkers> {
    let plain = plain_render.as_bytes();
    let chunked = chunked_render.as_bytes();

    if !contains_subslice(chunked, REASON_PROBE.as_bytes())
        || !contains_subslice(chunked, RESPONSE_PROBE.as_bytes())
    {
        return None;
    }

    let plain_size = plain.len();
    let chunked_size = chunked.len();
    let min_size = plain_size.min(chunked_size);

    let mut common_prefix = 0;
    while common_prefix < min_size && plain[common_prefix] == chunked[common_prefix] {
        common_prefix += 1;
    }

    let mut common_suffix = 0;
    while common_suffix < min_size - common_prefix
        && plain[plain_size - 1 - common_suffix] == chunked[chunked_size - 1 - common_suffix]
    {
        common_suffix += 1;
    }

    if common_prefix + common_suffix > chunked_size {
        return None;
    }

    let diff = &chunked[common_prefix..chunked_size - common_suffix];
    let reason_pos = find_subslice(diff, REASON_PROBE.as_bytes())?;

    let open = std::str::from_utf8(&diff[..reason_pos])
        .ok()?
        .trim()
        .to_owned();
    let close = std::str::from_utf8(&diff[reason_pos + REASON_PROBE.len()..])
        .ok()?
        .trim()
        .to_owned();

    if open.is_empty() || close.is_empty() {
        return None;
    }
    if open.contains(REASON_PROBE) || open.contains(RESPONSE_PROBE) {
        return None;
    }
    if close.contains(REASON_PROBE) || close.contains(RESPONSE_PROBE) {
        return None;
    }

    Some(ReasoningMarkers { open, close })
}

#[cfg(test)]
mod tests {
    use super::REASON_PROBE;
    use super::RESPONSE_PROBE;
    use super::extract_reasoning_markers_from_probe_renders;

    #[test]
    fn extracts_open_and_close_markers_from_diff() {
        let plain = format!("PREFIX{RESPONSE_PROBE}SUFFIX");
        let chunked = format!("PREFIX<think>{REASON_PROBE}</think>{RESPONSE_PROBE}SUFFIX");

        let markers = extract_reasoning_markers_from_probe_renders(&plain, &chunked)
            .expect("markers detected");

        assert_eq!(markers.open, "<think>");
        assert_eq!(markers.close, "</think>");
    }

    #[test]
    fn returns_none_when_chunked_render_lacks_probes() {
        let plain = "PREFIX-no-probe-SUFFIX";
        let chunked = "PREFIX-still-no-probe-SUFFIX";

        assert!(extract_reasoning_markers_from_probe_renders(plain, chunked).is_none());
    }

    #[test]
    fn returns_none_when_a_marker_would_be_empty() {
        let plain = format!("PREFIX{RESPONSE_PROBE}SUFFIX");
        let chunked = format!("PREFIX{REASON_PROBE}</think>{RESPONSE_PROBE}SUFFIX");

        assert!(extract_reasoning_markers_from_probe_renders(&plain, &chunked).is_none());
    }

    #[test]
    fn returns_none_when_marker_leaks_a_probe_sentinel() {
        let plain = format!("PREFIX{RESPONSE_PROBE}SUFFIX");
        let chunked =
            format!("PREFIX<think{RESPONSE_PROBE}>{REASON_PROBE}</think>{RESPONSE_PROBE}SUFFIX");

        assert!(extract_reasoning_markers_from_probe_renders(&plain, &chunked).is_none());
    }
}
