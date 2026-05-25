use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSeqStateError {
    #[error("Failed to save sequence state file")]
    FailedToSave,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}
