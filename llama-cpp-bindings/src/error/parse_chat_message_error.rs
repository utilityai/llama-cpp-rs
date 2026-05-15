use std::string::FromUtf8Error;

use crate::error::tool_call_format_failure::ToolCallFormatFailure;

/// Failed to parse a chat message via [`crate::Model::parse_chat_message`].
#[derive(Debug, thiserror::Error)]
pub enum ParseChatMessageError {
    /// llama.cpp returned an error code from the parse FFI call.
    #[error("ffi error {0}")]
    FfiError(i32),
    /// The C++ side threw an exception while parsing.
    #[error("c++ exception during chat parse: {0}")]
    ParseException(String),
    /// An accessor returned bytes that were not valid UTF-8.
    #[error("ffi returned non-utf8 string: {0}")]
    StringUtf8Error(#[from] FromUtf8Error),
    /// The caller passed a `tools_json` argument that is not valid JSON.
    #[error("tools_json is not valid JSON: {0}")]
    ToolsJsonInvalid(#[source] serde_json::Error),
    /// The caller passed a `tools_json` argument that parses as JSON but is not an array.
    #[error("tools_json must be a JSON array")]
    ToolsJsonNotArray,
    /// Failed to serialize the tools array for the FFI call.
    #[error("could not serialize tools to JSON: {0}")]
    ToolsSerialization(String),
    /// The model has no usable chat template, so the parser cannot be built.
    #[error("model has no chat template")]
    NoChatTemplate,
    /// The wrapper-side fallback parser detected a structural issue while parsing the body.
    #[error("template-override fallback parser failed: {0}")]
    TemplateOverrideFailed(#[from] ToolCallFormatFailure),
}
