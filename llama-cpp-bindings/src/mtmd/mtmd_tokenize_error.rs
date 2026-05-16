use crate::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;

#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    #[error("Failed to create CString from input text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("{0}")]
    InputChunksError(#[from] MtmdInputChunksError),
    #[error("Wrapper received a null mtmd-context argument")]
    NullCtxArg,
    #[error("Wrapper received a null output-chunks argument")]
    NullOutputArg,
    #[error("Wrapper received a null input-text argument")]
    NullTextArg,
    #[error("Wrapper received a null bitmaps argument with num_bitmaps > 0")]
    NullBitmapsArgWhenNumBitmapsNonzero,
    #[error("mtmd_tokenize reported that the number of bitmaps does not match the number of markers in the text")]
    BitmapCountDoesNotMatchMarkerCount,
    #[error("mtmd_tokenize reported an image preprocessing error")]
    ImagePreprocessingError,
    #[error("mtmd_tokenize returned an undocumented nonzero code: {code}")]
    VendoredReturnedUndocumentedNonzeroCode { code: i32 },
    #[error("Wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("mtmd_tokenize threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
