use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GgufContextError {
    #[error("Failed to initialize GGUF context from file: {0}")]
    InitFailed(PathBuf),

    #[error("Key not found in GGUF context: {key}")]
    KeyNotFound { key: String },

    #[error("null byte in string: {0}")]
    NulError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    #[error("GGUF value is not valid UTF-8: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}
