use std::os::raw::c_int;
use std::string::FromUtf8Error;

/// An error that can occur when converting a token to a string.
#[derive(Debug, thiserror::Error, Clone)]
#[non_exhaustive]
pub enum TokenToStringError {
    /// the token type was unknown
    #[error("Unknown Token Type")]
    UnknownTokenType,
    /// There was insufficient buffer space to convert the token to a string.
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    /// The token was not valid utf8.
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// An integer conversion failed.
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}
