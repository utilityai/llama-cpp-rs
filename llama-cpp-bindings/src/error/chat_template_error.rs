use std::ffi::NulError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateError {
    #[error("chat template not found - returned null pointer")]
    MissingTemplate,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}
