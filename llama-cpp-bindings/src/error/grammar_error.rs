use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("grammar root not found in grammar string")]
    RootNotFound,
    #[error("trigger word contains null bytes: {0}")]
    TriggerWordNullBytes(NulError),
    #[error("grammar string or root contains null bytes: {0}")]
    GrammarNullBytes(NulError),
    #[error("string contains null bytes: {0}")]
    NulError(#[from] NulError),
    #[error("integer overflow: {0}")]
    IntegerOverflow(String),
    #[error("llguidance error: {0}")]
    LlguidanceError(String),
    #[error("grammar is malformed")]
    GrammarMalformed,
    #[error("lazy grammar is malformed")]
    LazyGrammarMalformed,
    #[error("lazy-patterns grammar is malformed")]
    LazyPatternsGrammarMalformed,
    #[error("trigger pattern is not a valid regex: {message}")]
    InvalidTriggerPattern { message: String },
    #[error("llguidance sampler could not be created")]
    LlguidanceSamplerUnavailable,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
