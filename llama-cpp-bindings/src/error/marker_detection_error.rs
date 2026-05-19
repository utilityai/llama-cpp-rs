use std::string::FromUtf8Error;

#[derive(Debug, thiserror::Error)]
pub enum MarkerDetectionError {
    #[error("ffi returned non-utf8 marker bytes: {0}")]
    MarkerUtf8Error(#[from] FromUtf8Error),
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("reasoning-marker detection failed: {message}")]
    ReasoningMarkerDetectionFailed { message: String },
    #[error("tool-call haystack computation failed: {message}")]
    ToolCallHaystackComputationFailed { message: String },
    #[error("tool-call synthetic-render diagnosis failed: {message}")]
    ToolCallSyntheticRenderDiagnosisFailed { message: String },
}
