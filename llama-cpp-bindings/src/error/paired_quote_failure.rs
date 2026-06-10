#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum PairedQuoteFailure {
    #[error("empty key in tool call '{tool_name}' arguments")]
    EmptyKey { tool_name: String },
    #[error("tool call '{tool_name}' translated arguments are not valid JSON: {message}")]
    InvalidJsonArguments { tool_name: String, message: String },
    #[error("tool call '{tool_name}' has unclosed quoted value for key '{key}'")]
    UnclosedQuotedValue { tool_name: String, key: String },
    #[error("tool call '{tool_name}' arguments ended without close marker (state: {state})")]
    UnclosedArgumentBlock {
        tool_name: String,
        state: &'static str,
    },
    #[error(
        "tool call '{tool_name}' has unexpected character '{character}' after value for key '{key}'"
    )]
    UnexpectedCharAfterValue {
        tool_name: String,
        key: String,
        character: char,
    },
}
