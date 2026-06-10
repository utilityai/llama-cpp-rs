use std::path::PathBuf;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdInitError {
    #[error("Failed to create CString from mmproj path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Mmproj path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("mmproj could not be loaded: {path:?}")]
    Unloadable { path: PathBuf },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
