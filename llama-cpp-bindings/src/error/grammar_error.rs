use std::ffi::NulError;

use crate::error::token_to_string_error::TokenToStringError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GrammarError {
    #[error("the approximate token environment could not be built: {0}")]
    TokEnvUnavailable(#[from] TokenToStringError),
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
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}
