/// Effective tool-call marker strings resolved from either the autoparser
/// output or the per-template override registry.
///
/// Each side is independently optional because the autoparser may report only
/// one of the two strings, and the override registry may not match the
/// template at all.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedToolCallMarkers {
    pub open: Option<String>,
    pub close: Option<String>,
}
