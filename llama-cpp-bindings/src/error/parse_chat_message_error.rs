use std::string::FromUtf8Error;

use crate::error::tool_call_format_failure::ToolCallFormatFailure;

#[derive(Debug, thiserror::Error)]
pub enum ParseChatMessageError {
    #[error("llama_rs_parse_chat_message called with null model")]
    ParseNullModelArg,
    #[error("llama_rs_parse_chat_message called with null input")]
    ParseNullInputArg,
    #[error("llama_rs_parse_chat_message called with null out_handle")]
    ParseNullOutHandleArg,
    #[error("llama_rs_parse_chat_message called with null out_error")]
    ParseNullOutErrorArg,
    #[error("model has no chat template")]
    ParseModelHasNoChatTemplate,
    #[error("model has no vocab")]
    ParseModelHasNoVocab,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ParseErrorStringAllocationFailed,
    #[error("c++ exception during chat parse: {message}")]
    ParseException { message: String },
    #[error("llama_rs_parsed_chat_free destructor threw a C++ exception: {message}")]
    FreeDestructorThrewCxxException { message: String },
    #[error("llama_rs_parsed_chat_free wrapper failed to duplicate the C++ exception string")]
    FreeErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_tool_call_count called with null handle")]
    ToolCallCountNullHandleArg,
    #[error("llama_rs_parsed_chat_tool_call_count threw a C++ exception: {message}")]
    ToolCallCountThrewCxxException { message: String },
    #[error("llama_rs_parsed_chat_tool_call_count wrapper failed to duplicate the C++ exception string")]
    ToolCallCountErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_tool_call_id called with null handle")]
    ToolCallIdNullHandleArg,
    #[error("llama_rs_parsed_chat_tool_call_id called with index {index} out of bounds")]
    ToolCallIdIndexOutOfBounds { index: usize },
    #[error("llama_rs_parsed_chat_tool_call_id threw a C++ exception: {message}")]
    ToolCallIdThrewCxxException { message: String },
    #[error("llama_rs_parsed_chat_tool_call_id wrapper failed to duplicate the C++ exception string")]
    ToolCallIdErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_tool_call_name called with null handle")]
    ToolCallNameNullHandleArg,
    #[error("llama_rs_parsed_chat_tool_call_name called with index {index} out of bounds")]
    ToolCallNameIndexOutOfBounds { index: usize },
    #[error("llama_rs_parsed_chat_tool_call_name threw a C++ exception: {message}")]
    ToolCallNameThrewCxxException { message: String },
    #[error("llama_rs_parsed_chat_tool_call_name wrapper failed to duplicate the C++ exception string")]
    ToolCallNameErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_tool_call_arguments called with null handle")]
    ToolCallArgumentsNullHandleArg,
    #[error("llama_rs_parsed_chat_tool_call_arguments called with index {index} out of bounds")]
    ToolCallArgumentsIndexOutOfBounds { index: usize },
    #[error("llama_rs_parsed_chat_tool_call_arguments threw a C++ exception: {message}")]
    ToolCallArgumentsThrewCxxException { message: String },
    #[error(
        "llama_rs_parsed_chat_tool_call_arguments wrapper failed to duplicate the C++ exception string"
    )]
    ToolCallArgumentsErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_content called with null handle")]
    ContentNullHandleArg,
    #[error("llama_rs_parsed_chat_content threw a C++ exception: {message}")]
    ContentThrewCxxException { message: String },
    #[error("llama_rs_parsed_chat_content wrapper failed to duplicate the C++ exception string")]
    ContentErrorStringAllocationFailed,
    #[error("llama_rs_parsed_chat_reasoning_content called with null handle")]
    ReasoningContentNullHandleArg,
    #[error("llama_rs_parsed_chat_reasoning_content threw a C++ exception: {message}")]
    ReasoningContentThrewCxxException { message: String },
    #[error(
        "llama_rs_parsed_chat_reasoning_content wrapper failed to duplicate the C++ exception string"
    )]
    ReasoningContentErrorStringAllocationFailed,
    #[error("ffi returned non-utf8 string: {0}")]
    StringUtf8Error(#[from] FromUtf8Error),
    #[error("tools_json is not valid JSON: {0}")]
    ToolsJsonInvalid(#[source] serde_json::Error),
    #[error("tools_json must be a JSON array")]
    ToolsJsonNotArray,
    #[error("could not serialize tools to JSON: {0}")]
    ToolsSerialization(String),
    #[error("template-override fallback parser failed: {0}")]
    TemplateOverrideFailed(#[from] ToolCallFormatFailure),
}
