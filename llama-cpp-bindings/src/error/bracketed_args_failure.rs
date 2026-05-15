/// Failures specific to the bracketed-JSON args parser (Mistral 3 `[TOOL_CALLS]name[ARGS]{...}`).
#[derive(Debug, thiserror::Error)]
pub enum BracketedArgsFailure {
    #[error("tool call '{tool_name}' arguments are not valid JSON: {message}")]
    InvalidJsonArguments { tool_name: String, message: String },
    #[error("tool call '{tool_name}' arguments truncated before JSON value completed")]
    UnterminatedArguments { tool_name: String },
}
