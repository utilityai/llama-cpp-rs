/// Errors that can occur when initializing MTMD context
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// MTMD context initialization returned null
    #[error("MTMD context initialization returned null")]
    NullResult,
}

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

/// Errors that can occur when working with MTMD input chunks collections
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// Input chunks creation returned null
    #[error("Input chunks creation returned null")]
    NullResult,
}

/// Errors that can occur when working with individual MTMD input chunks
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// Input chunk operation returned null
    #[error("Input chunk operation returned null")]
    NullResult,
}

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

/// Errors that can occur during encoding
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// Encode operation failed
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}

use crate::mtmd::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;

/// Errors that can occur during evaluation
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// Requested batch size exceeds the context's maximum batch size
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit {
        /// The batch size requested in `eval_chunks`
        requested: i32,
        /// The maximum batch size configured on the context
        context_max: u32,
    },
    /// An image chunk's token count exceeds the per-decode `n_batch` budget,
    /// so handing it to `llama_decode` would trip the GGML_ASSERT.
    #[error(
        "image chunk has {} tokens but n_batch is {}",
        .0.image_tokens,
        .0.n_batch,
    )]
    ImageChunkExceedsBatchSize(ImageChunkBatchSizeMismatch),
    /// Evaluation operation failed
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}
