#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSplitModeParseError {
    pub value: i32,
    pub context: String,
}
