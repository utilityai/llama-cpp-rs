use std::string::FromUtf8Error;

#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}
