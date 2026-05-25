use std::ffi::NulError;
use std::string::FromUtf8Error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MetaValError {
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),

    #[error("Negative return value. Likely due to a missing index or key. Got return value: {0}")]
    NegativeReturn(i32),
}
