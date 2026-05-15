/// An error that occurs when unknown split mode is encountered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSplitModeParseError {
    /// The value that could not be parsed as a split mode.
    pub value: i32,
    /// Additional context about why the parse failed.
    pub context: String,
}
