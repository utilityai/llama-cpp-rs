use std::ffi::NulError;
use std::path::PathBuf;

/// Error returned when loading GGML backend modules from a path.
#[derive(Debug, thiserror::Error)]
pub enum LoadBackendsError {
    /// The provided path could not be converted to UTF-8.
    #[error("backend directory path is not valid UTF-8: {0}")]
    PathNotUtf8(PathBuf),
    /// The provided path contained an interior null byte.
    #[error("backend directory path contains a null byte: {0}")]
    PathNullByte(#[from] NulError),
}
