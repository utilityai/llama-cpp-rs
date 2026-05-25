#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    #[error("Empty buffer")]
    EmptyBuffer,
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
}
