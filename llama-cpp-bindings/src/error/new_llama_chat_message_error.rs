use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    #[error("{0}")]
    NulError(#[from] NulError),
}
