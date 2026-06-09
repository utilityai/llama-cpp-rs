#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSplitModeParseError {
    pub value: u32,
    pub context: String,
}
