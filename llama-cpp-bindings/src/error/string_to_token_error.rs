use std::ffi::NulError;

/// Failed to convert a string to a token sequence.
#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    /// Failed to convert a provided integer to a [`c_int`].
    CIntConversionError(#[from] std::num::TryFromIntError),
}
