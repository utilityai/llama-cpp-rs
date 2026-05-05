use crate::token::LlamaToken;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SampledToken {
    Content(LlamaToken),
    Reasoning(LlamaToken),
    ToolCall(LlamaToken),
    Undeterminable(LlamaToken),
}
