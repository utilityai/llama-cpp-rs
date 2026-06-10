use std::str::Utf8Error;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum MtmdDefaultMarkerError {
    #[error("llama.cpp mtmd_default_marker returned bytes that are not valid UTF-8: {0}")]
    NotUtf8(#[from] Utf8Error),
}
