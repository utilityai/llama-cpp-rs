use std::ffi::NulError;

/// Errors that can occur when initializing a grammar sampler
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GrammarError {
    /// The grammar root was not found in the grammar string
    #[error("Grammar root not found in grammar string")]
    RootNotFound,
    /// The trigger word contains null bytes
    #[error("Trigger word contains null bytes: {0}")]
    TriggerWordNullBytes(NulError),
    /// The grammar string or root contains null bytes
    #[error("Grammar string or root contains null bytes: {0}")]
    GrammarNullBytes(NulError),
    /// A string contains null bytes
    #[error("String contains null bytes: {0}")]
    NulError(#[from] NulError),
    /// The grammar call returned null
    #[error("Grammar initialization failed: {0}")]
    NullGrammar(String),
    /// An integer value exceeded the allowed range
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
    /// An error from the llguidance library
    #[error("llguidance error: {0}")]
    LlguidanceError(String),
}
