use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    #[error("Failed to save session file")]
    FailedToSave,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}
