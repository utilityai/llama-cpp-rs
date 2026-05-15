/// When logits-related functions fail
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LogitsError {
    /// The logits data pointer is null.
    #[error("logits data pointer is null")]
    NullLogits,
    /// The requested token index has not been initialized for logits.
    #[error("logit for token index {0} is not initialized")]
    TokenNotInitialized(i32),
    /// The token index exceeds the context size.
    #[error("token index {token_index} exceeds context size {context_size}")]
    TokenIndexExceedsContext {
        /// The token index that was requested.
        token_index: u32,
        /// The context size.
        context_size: u32,
    },
    /// The vocabulary size does not fit into a usize.
    #[error("n_vocab does not fit into usize: {0}")]
    VocabSizeOverflow(#[source] std::num::TryFromIntError),
    /// The token index does not fit into a u32.
    #[error("token_index does not fit into u32: {0}")]
    TokenIndexOverflow(#[source] std::num::TryFromIntError),
}
