/// Errors that can occur when working with MTMD bitmaps
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Invalid data size for bitmap
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    /// Image dimensions too small for processing (minimum 2x2)
    #[error("Image dimensions too small: {0}x{1} (minimum 2x2)")]
    ImageDimensionsTooSmall(u32, u32),
    /// Bitmap creation returned null
    #[error("Bitmap creation returned null")]
    NullResult,
}
