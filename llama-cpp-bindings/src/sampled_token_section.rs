#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SampledTokenSection {
    Pending,
    Content,
    Reasoning,
    ToolCall,
}
