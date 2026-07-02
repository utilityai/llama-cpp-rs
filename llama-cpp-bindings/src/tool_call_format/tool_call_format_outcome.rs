use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;

use crate::error::tool_call_format_failure::ToolCallFormatFailure;

#[derive(Debug, Eq, PartialEq)]
pub enum ToolCallFormatOutcome {
    Parsed(Vec<ParsedToolCall>),
    NoMatch,
    Failed(ToolCallFormatFailure),
}
