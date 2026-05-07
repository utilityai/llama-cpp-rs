use llama_cpp_bindings_types::ParsedToolCall;

use crate::error::ToolCallFormatFailure;

#[derive(Debug)]
pub enum ToolCallFormatOutcome {
    Parsed(Vec<ParsedToolCall>),
    NoMatch,
    Failed(ToolCallFormatFailure),
}
