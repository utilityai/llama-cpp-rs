use std::str::Utf8Error;
use std::string::FromUtf8Error;

use crate::error::chat_template_error::ChatTemplateError;
use crate::error::string_to_token_error::StringToTokenError;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
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
    #[error("a detected marker string could not be tokenised: {0}")]
    MarkerTokenizationFailed(#[from] StringToTokenError),
    #[error("the chat template is not valid UTF-8: {0}")]
    ToolCallTemplateNotUtf8(#[from] Utf8Error),
    #[error("the chat template could not be retrieved for tool-call marker detection: {0}")]
    ChatTemplateUnavailable(#[source] ChatTemplateError),
}
