use std::string::FromUtf8Error;

/// Failed to detect tool-call diagnostic markers for a model.
#[derive(Debug, thiserror::Error)]
pub enum MarkerDetectionError {
    /// llama.cpp returned an error code from the marker detection FFI call.
    #[error("ffi error {0}")]
    FfiError(i32),
    /// The C++ side threw an exception during template analysis.
    #[error("c++ exception during template analysis: {0}")]
    AnalyzeException(String),
    /// llama.cpp returned a marker string but its bytes were not valid UTF-8.
    #[error("ffi returned non-utf8 marker bytes: {0}")]
    MarkerUtf8Error(#[from] FromUtf8Error),
}
