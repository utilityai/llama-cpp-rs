use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterInitError {
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    #[error("adapter could not be loaded")]
    Unloadable,
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    #[error("adapter file not found: {0}")]
    FileNotFound(PathBuf),
}
