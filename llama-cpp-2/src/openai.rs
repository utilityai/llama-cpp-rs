//! OpenAI-compatible chat message parsing helpers.

use std::ffi::{CStr, CString};
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use serde_json::Value;

use crate::ChatParseError;

/// Parsed tool call extracted from model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIToolCall {
    /// Tool name.
    pub name: String,
    /// Tool arguments as JSON text.
    pub arguments: String,
    /// Tool call id.
    pub id: String,
}

/// Parsed chat message compatible with OpenAI-style schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIChatMessage {
    /// Message role.
    pub role: String,
    /// Message content.
    pub content: Option<String>,
    /// Optional reasoning content.
    pub reasoning_content: Option<String>,
    /// Optional tool name (for tool responses).
    pub tool_name: Option<String>,
    /// Optional tool call id (for tool responses).
    pub tool_call_id: Option<String>,
    /// Parsed tool calls.
    pub tool_calls: Vec<OpenAIToolCall>,
}

impl OpenAIChatMessage {
    fn from_raw_msg(raw_msg: &llama_cpp_sys_2::llama_rs_chat_msg) -> Result<Self, ChatParseError> {
        let role = if raw_msg.role.is_null() {
            "assistant".to_string()
        } else {
            let bytes = unsafe { CStr::from_ptr(raw_msg.role) }.to_bytes().to_vec();
            let role = String::from_utf8(bytes)?;
            if role.is_empty() {
                "assistant".to_string()
            } else {
                role
            }
        };

        let content = if raw_msg.content.is_null() {
            None
        } else {
            let bytes = unsafe { CStr::from_ptr(raw_msg.content) }.to_bytes().to_vec();
            Some(String::from_utf8(bytes)?)
        };
        let reasoning_content = if raw_msg.reasoning_content.is_null() {
            None
        } else {
            let bytes = unsafe { CStr::from_ptr(raw_msg.reasoning_content) }
                .to_bytes()
                .to_vec();
            Some(String::from_utf8(bytes)?)
        };
        let tool_name = if raw_msg.tool_name.is_null() {
            None
        } else {
            let bytes = unsafe { CStr::from_ptr(raw_msg.tool_name) }
                .to_bytes()
                .to_vec();
            Some(String::from_utf8(bytes)?)
        };
        let tool_call_id = if raw_msg.tool_call_id.is_null() {
            None
        } else {
            let bytes = unsafe { CStr::from_ptr(raw_msg.tool_call_id) }
                .to_bytes()
                .to_vec();
            Some(String::from_utf8(bytes)?)
        };

        let mut tool_calls = Vec::new();
        if raw_msg.tool_calls_count > 0 {
            if raw_msg.tool_calls.is_null() {
                return Err(ChatParseError::NullResult);
            }
            let calls = unsafe { slice::from_raw_parts(raw_msg.tool_calls, raw_msg.tool_calls_count) };
            tool_calls.reserve(calls.len());
            for call in calls {
                let name = if call.name.is_null() {
                    String::new()
                } else {
                    let bytes = unsafe { CStr::from_ptr(call.name) }.to_bytes().to_vec();
                    String::from_utf8(bytes)?
                };
                let arguments = if call.arguments.is_null() {
                    String::new()
                } else {
                    let bytes = unsafe { CStr::from_ptr(call.arguments) }.to_bytes().to_vec();
                    String::from_utf8(bytes)?
                };
                let id = if call.id.is_null() {
                    String::new()
                } else {
                    let bytes = unsafe { CStr::from_ptr(call.id) }.to_bytes().to_vec();
                    String::from_utf8(bytes)?
                };
                tool_calls.push(OpenAIToolCall { name, arguments, id });
            }
        }

        Ok(OpenAIChatMessage {
            role,
            content,
            reasoning_content,
            tool_name,
            tool_call_id,
            tool_calls,
        })
    }

    /// Parse a generated response using llama.cpp's OpenAI-compatible parser.
    pub fn parse_from_llama(
        text: &str,
        is_partial: bool,
        chat_format: i32,
        parse_tool_calls: bool,
        parser: Option<&str>,
        thinking_forced_open: bool,
    ) -> Result<Self, ChatParseError> {
        let text_cstr = CString::new(text)?;
        let parser_cstr = parser.map(CString::new).transpose()?;

        let mut raw_msg = llama_cpp_sys_2::llama_rs_chat_msg {
            role: ptr::null_mut(),
            content: ptr::null_mut(),
            reasoning_content: ptr::null_mut(),
            tool_name: ptr::null_mut(),
            tool_call_id: ptr::null_mut(),
            tool_calls: ptr::null_mut(),
            tool_calls_count: 0,
        };

        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parse(
                text_cstr.as_ptr(),
                is_partial,
                chat_format,
                parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                thinking_forced_open,
                &mut raw_msg,
            )
        };

        let result = (|| {
            if rc != 0 {
                return Err(ChatParseError::FfiError(rc));
            }
            OpenAIChatMessage::from_raw_msg(&raw_msg)
        })();

        unsafe { llama_cpp_sys_2::llama_rs_chat_msg_free(&mut raw_msg) };
        result
    }

    /// Convert the parsed message into an OpenAI-compatible JSON object.
    #[must_use]
    pub fn to_oaicompat_value(&self) -> Value {
        let mut message = serde_json::Map::new();
        let role = if self.role.is_empty() {
            "assistant".to_string()
        } else {
            self.role.clone()
        };
        message.insert("role".to_string(), Value::String(role));

        if let Some(reasoning) = &self.reasoning_content {
            message.insert("reasoning_content".to_string(), Value::String(reasoning.clone()));
        }
        if let Some(name) = &self.tool_name {
            message.insert("name".to_string(), Value::String(name.clone()));
        }
        if let Some(tool_call_id) = &self.tool_call_id {
            message.insert(
                "tool_call_id".to_string(),
                Value::String(tool_call_id.clone()),
            );
        }

        if !self.tool_calls.is_empty() {
            let calls = self
                .tool_calls
                .iter()
                .map(|call| {
                    let mut func = serde_json::Map::new();
                    func.insert("name".to_string(), Value::String(call.name.clone()));
                    func.insert("arguments".to_string(), Value::String(call.arguments.clone()));

                    let mut entry = serde_json::Map::new();
                    entry.insert("type".to_string(), Value::String("function".to_string()));
                    entry.insert("function".to_string(), Value::Object(func));
                    entry.insert("id".to_string(), Value::String(call.id.clone()));
                    Value::Object(entry)
                })
                .collect::<Vec<_>>();
            message.insert("tool_calls".to_string(), Value::Array(calls));
        }

        let content_value = if !self.tool_calls.is_empty()
            && self.content.as_deref().unwrap_or("").is_empty()
        {
            Value::Null
        } else {
            Value::String(self.content.clone().unwrap_or_default())
        };
        message.insert("content".to_string(), content_value);

        Value::Object(message)
    }
}

/// Partial message delta produced while streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIChatDelta {
    /// Optional reasoning delta.
    pub reasoning_content_delta: Option<String>,
    /// Optional content delta.
    pub content_delta: Option<String>,
    /// Optional tool call delta.
    pub tool_call: Option<OpenAIToolCallDelta>,
}

/// Tool call delta for streaming updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIToolCallDelta {
    /// Tool call index.
    pub index: usize,
    /// Optional tool call id.
    pub id: Option<String>,
    /// Optional tool name.
    pub name: Option<String>,
    /// Tool arguments delta.
    pub arguments: String,
}

impl OpenAIChatDelta {
    /// Convert the delta into an OpenAI-compatible `delta` JSON object.
    #[must_use]
    pub fn to_oaicompat_value(&self) -> Value {
        let mut delta = serde_json::Map::new();
        if let Some(reasoning) = &self.reasoning_content_delta {
            delta.insert("reasoning_content".to_string(), Value::String(reasoning.clone()));
        }
        if let Some(content) = &self.content_delta {
            delta.insert("content".to_string(), Value::String(content.clone()));
        }
        if let Some(tool_call) = &self.tool_call {
            let mut tool_call_obj = serde_json::Map::new();
            tool_call_obj.insert(
                "index".to_string(),
                Value::Number(serde_json::Number::from(tool_call.index as u64)),
            );
            if let Some(id) = &tool_call.id {
                if !id.is_empty() {
                    tool_call_obj.insert("id".to_string(), Value::String(id.clone()));
                    tool_call_obj.insert(
                        "type".to_string(),
                        Value::String("function".to_string()),
                    );
                }
            }
            let mut function = serde_json::Map::new();
            if let Some(name) = &tool_call.name {
                if !name.is_empty() {
                    function.insert("name".to_string(), Value::String(name.clone()));
                }
            }
            function.insert(
                "arguments".to_string(),
                Value::String(tool_call.arguments.clone()),
            );
            tool_call_obj.insert("function".to_string(), Value::Object(function));
            delta.insert(
                "tool_calls".to_string(),
                Value::Array(vec![Value::Object(tool_call_obj)]),
            );
        }
        Value::Object(delta)
    }
}

/// Parsed streaming update from incremental text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIChatStreamUpdate {
    /// Current parsed message.
    pub message: OpenAIChatMessage,
    /// Deltas generated by this update.
    pub deltas: Vec<OpenAIChatDelta>,
}

/// Stateful streaming parser matching llama.cpp server behavior.
#[derive(Debug)]
pub struct OpenAIChatStreamParser {
    state: NonNull<llama_cpp_sys_2::llama_rs_chat_parse_state>,
}

impl OpenAIChatStreamParser {
    /// Create a new streaming parser.
    pub fn new(
        chat_format: i32,
        parse_tool_calls: bool,
        parser: Option<&str>,
        thinking_forced_open: bool,
    ) -> Result<Self, ChatParseError> {
        let parser_cstr = parser.map(CString::new).transpose()?;
        let state = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parse_state_init(
                chat_format,
                parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                thinking_forced_open,
            )
        };
        let state = NonNull::new(state).ok_or(ChatParseError::NullResult)?;
        Ok(Self { state })
    }

    /// Append new text and return streaming deltas.
    pub fn update(
        &mut self,
        text_added: &str,
        is_partial: bool,
    ) -> Result<OpenAIChatStreamUpdate, ChatParseError> {
        let text_cstr = CString::new(text_added)?;
        let mut raw_msg = llama_cpp_sys_2::llama_rs_chat_msg {
            role: ptr::null_mut(),
            content: ptr::null_mut(),
            reasoning_content: ptr::null_mut(),
            tool_name: ptr::null_mut(),
            tool_call_id: ptr::null_mut(),
            tool_calls: ptr::null_mut(),
            tool_calls_count: 0,
        };
        let mut diffs_ptr: *mut llama_cpp_sys_2::llama_rs_chat_msg_diff = ptr::null_mut();
        let mut diffs_count: usize = 0;

        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parse_state_update(
                self.state.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &mut raw_msg,
                &mut diffs_ptr,
                &mut diffs_count,
            )
        };

        let result = (|| {
            if rc != 0 {
                return Err(ChatParseError::FfiError(rc));
            }
            let message = OpenAIChatMessage::from_raw_msg(&raw_msg)?;

            let mut deltas = Vec::new();
            if diffs_count > 0 {
                if diffs_ptr.is_null() {
                    return Err(ChatParseError::NullResult);
                }
                let diffs = unsafe { slice::from_raw_parts(diffs_ptr, diffs_count) };
                deltas.reserve(diffs.len());
                for diff in diffs {
                    let reasoning_content_delta = if diff.reasoning_content_delta.is_null() {
                        None
                    } else {
                        let bytes =
                            unsafe { CStr::from_ptr(diff.reasoning_content_delta) }
                                .to_bytes()
                                .to_vec();
                        let delta = String::from_utf8(bytes)?;
                        if delta.is_empty() { None } else { Some(delta) }
                    };
                    let content_delta = if diff.content_delta.is_null() {
                        None
                    } else {
                        let bytes =
                            unsafe { CStr::from_ptr(diff.content_delta) }
                                .to_bytes()
                                .to_vec();
                        let delta = String::from_utf8(bytes)?;
                        if delta.is_empty() { None } else { Some(delta) }
                    };
                    let tool_call = if diff.tool_call_index == usize::MAX {
                        None
                    } else {
                        let name = if diff.tool_call_delta.name.is_null() {
                            None
                        } else {
                            let bytes = unsafe { CStr::from_ptr(diff.tool_call_delta.name) }
                                .to_bytes()
                                .to_vec();
                            let value = String::from_utf8(bytes)?;
                            if value.is_empty() { None } else { Some(value) }
                        };
                        let arguments = if diff.tool_call_delta.arguments.is_null() {
                            String::new()
                        } else {
                            let bytes = unsafe { CStr::from_ptr(diff.tool_call_delta.arguments) }
                                .to_bytes()
                                .to_vec();
                            String::from_utf8(bytes)?
                        };
                        let id = if diff.tool_call_delta.id.is_null() {
                            None
                        } else {
                            let bytes = unsafe { CStr::from_ptr(diff.tool_call_delta.id) }
                                .to_bytes()
                                .to_vec();
                            let value = String::from_utf8(bytes)?;
                            if value.is_empty() { None } else { Some(value) }
                        };
                        Some(OpenAIToolCallDelta {
                            index: diff.tool_call_index,
                            id,
                            name,
                            arguments,
                        })
                    };

                    deltas.push(OpenAIChatDelta {
                        reasoning_content_delta,
                        content_delta,
                        tool_call,
                    });
                }
            }

            Ok(OpenAIChatStreamUpdate { message, deltas })
        })();

        unsafe {
            llama_cpp_sys_2::llama_rs_chat_msg_free(&mut raw_msg);
            llama_cpp_sys_2::llama_rs_chat_msg_diff_free(diffs_ptr, diffs_count);
        }
        result
    }
}

impl Drop for OpenAIChatStreamParser {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_chat_parse_state_free(self.state.as_ptr()) }
    }
}
