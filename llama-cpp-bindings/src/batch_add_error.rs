/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    /// Empty buffer is provided for [`crate::llama_batch::LlamaBatch::get_one`]
    #[error("Empty buffer")]
    EmptyBuffer,
    /// An integer value exceeded the allowed range.
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
}
