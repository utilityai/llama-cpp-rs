use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSessionError {
    #[error("Failed to load session file")]
    FailedToLoad,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    #[error("max_length is not large enough to hold {n_out} (was {max_tokens})")]
    InsufficientMaxLength { n_out: usize, max_tokens: usize },
}
