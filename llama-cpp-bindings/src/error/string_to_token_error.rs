use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    CIntConversionError(#[from] std::num::TryFromIntError),
    #[error("llama_rs_tokenize called with null vocab")]
    NullVocabArg,
    #[error("llama_rs_tokenize called with null text")]
    NullTextArg,
    #[error("llama_rs_tokenize called with null out_returned_count")]
    NullOutReturnedCountArg,
    #[error("llama_rs_tokenize called with null out_error")]
    NullOutErrorArg,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_tokenize threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
