use std::string::FromUtf8Error;

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    /// the string could not be converted to utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// An integer conversion failed.
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}
