#[derive(Debug, thiserror::Error)]
pub enum LlamaContextLoadError {
    #[error("context could not be constructed")]
    Unconstructible,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
