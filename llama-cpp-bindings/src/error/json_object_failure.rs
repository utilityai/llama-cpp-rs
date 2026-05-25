#[derive(Debug, thiserror::Error)]
pub enum JsonObjectFailure {
    #[error("tool call body has malformed JSON: {message}")]
    InvalidJson { message: String },
}
