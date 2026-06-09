use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum LlamaModelLoadError {
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    #[error("model file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("model could not be loaded")]
    Unloadable,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
