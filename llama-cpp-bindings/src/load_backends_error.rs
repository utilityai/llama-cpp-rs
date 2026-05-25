use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum LoadBackendsError {
    #[error("backend directory path is not valid UTF-8: {0}")]
    PathNotUtf8(PathBuf),
    #[error("backend directory path contains a null byte: {0}")]
    PathNullByte(#[from] NulError),
}
