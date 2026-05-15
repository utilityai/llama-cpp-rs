use crate::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;

/// Errors that can occur during tokenization
#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    /// Number of bitmaps does not match number of markers in text
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    /// Image preprocessing error occurred
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    /// Failed to create input chunks collection
    #[error("{0}")]
    InputChunksError(#[from] MtmdInputChunksError),
    /// Text contains characters that cannot be converted to C string
    #[error("Failed to create CString from text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// Unknown error occurred during tokenization
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}
