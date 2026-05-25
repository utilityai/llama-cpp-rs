#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedToolCallMarkers {
    pub open: Option<String>,
    pub close: Option<String>,
}
