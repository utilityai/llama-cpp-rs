/// Failures specific to the JSON-object args parser (Qwen 3 `<tool_call>{"name":..., "arguments":...}</tool_call>`).
#[derive(Debug, thiserror::Error)]
pub enum JsonObjectFailure {
    #[error("tool call body has malformed JSON: {message}")]
    InvalidJson { message: String },
}
