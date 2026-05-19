#[derive(Debug, thiserror::Error)]
pub enum SamplerAcceptError {
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("grammar state corrupted during accept: {message}")]
    GrammarStateCorrupted { message: String },
}
