use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    #[error("Failed to create CString from bitmap-source path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Bitmap-source path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    #[error("Image dimensions too small: {0}x{1} (minimum 2x2)")]
    ImageDimensionsTooSmall(u32, u32),
    #[error("mtmd_bitmap_init / mtmd_bitmap_init_from_audio returned null")]
    NullResult,
    #[error("Internal wrapper invariant violated: caller did not pass an out-bitmap pointer")]
    NullOutBitmapArg,
    #[error("Wrapper received a null mtmd-context argument")]
    NullCtxArg,
    #[error("Wrapper received a null bitmap-source-path argument")]
    NullFnameArg,
    #[error("mtmd_helper_bitmap_init_from_file returned null without throwing for path: {path:?}")]
    VendoredReturnedNull { path: PathBuf },
    #[error("Wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("mtmd_helper_bitmap_init_from_file threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
