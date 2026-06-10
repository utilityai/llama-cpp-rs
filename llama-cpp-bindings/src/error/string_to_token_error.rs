use std::ffi::NulError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum StringToTokenError {
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    CIntConversionError(#[from] std::num::TryFromIntError),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
