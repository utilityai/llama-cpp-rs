#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MarkerKind {
    ReasoningOpen,
    ReasoningClose,
    ToolCallOpen,
    ToolCallClose,
}
