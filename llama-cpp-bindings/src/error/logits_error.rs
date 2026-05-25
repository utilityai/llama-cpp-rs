#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LogitsError {
    #[error("logits data pointer is null")]
    NullLogits,
    #[error("logit for token index {0} is not initialized")]
    TokenNotInitialized(i32),
    #[error("token index {token_index} exceeds context size {context_size}")]
    TokenIndexExceedsContext { token_index: u32, context_size: u32 },
    #[error("n_vocab does not fit into usize: {0}")]
    VocabSizeOverflow(#[source] std::num::TryFromIntError),
    #[error("token_index does not fit into u32: {0}")]
    TokenIndexOverflow(#[source] std::num::TryFromIntError),
}
