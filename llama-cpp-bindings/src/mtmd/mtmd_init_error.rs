use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    #[error("Failed to create CString from mmproj path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Mmproj path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("Internal wrapper invariant violated: caller did not pass an out-ctx pointer")]
    NullOutCtxArg,
    #[error("Wrapper received a null mmproj-path argument")]
    NullMmprojPathArg,
    #[error("Wrapper received a null text-model argument")]
    NullTextModelArg,
    #[error("mtmd_init_from_file returned null without throwing for mmproj path: {path:?}")]
    VendoredReturnedNull { path: PathBuf },
    #[error("Wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("mtmd_init_from_file threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
