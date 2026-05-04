use std::ffi::{CStr, CString, c_char};
use std::ptr::{self, NonNull};

use serde::Deserialize;

use crate::model::grammar_trigger::{GrammarTrigger, GrammarTriggerType};
use crate::openai::ChatParseStateOaicompat;
use crate::token::LlamaToken;
use crate::{ApplyChatTemplateError, ChatParseError, status_is_ok, status_to_i32};

/// Result of applying a chat template with tool grammar support.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTemplateResult {
    /// Rendered chat prompt.
    pub prompt: String,
    /// Optional grammar generated from tool definitions.
    pub grammar: Option<String>,
    /// Whether to use lazy grammar sampling.
    pub grammar_lazy: bool,
    /// Lazy grammar triggers derived from the template.
    pub grammar_triggers: Vec<GrammarTrigger>,
    /// Tokens that should be preserved for sampling.
    pub preserved_tokens: Vec<String>,
    /// Additional stop sequences added by the template.
    pub additional_stops: Vec<String>,
    /// Chat format used for parsing responses.
    pub chat_format: i32,
    /// Optional serialized PEG parser for tool-call parsing.
    pub parser: Option<String>,
    /// Generation prompt prefix that was prepended to the conversation.
    /// Empty when the template did not produce one. The parser uses this prefix to
    /// reconstruct responses without misattributing model output to the prompt.
    pub generation_prompt: String,
    /// Whether the model supports thinking/reasoning blocks.
    pub supports_thinking: bool,
    /// Whether tool calls should be parsed from the response.
    pub parse_tool_calls: bool,
}

#[derive(Deserialize)]
struct RawTemplateResult {
    prompt: String,
    chat_format: i32,
    supports_thinking: bool,
    grammar_lazy: bool,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    parser: Option<String>,
    #[serde(default)]
    generation_prompt: String,
    grammar_triggers: Vec<RawGrammarTrigger>,
    preserved_tokens: Vec<String>,
    additional_stops: Vec<String>,
}

#[derive(Deserialize)]
struct RawGrammarTrigger {
    #[serde(rename = "type")]
    type_: i32,
    value: String,
    token: i32,
}

impl TryFrom<RawGrammarTrigger> for GrammarTrigger {
    type Error = ApplyChatTemplateError;

    fn try_from(raw: RawGrammarTrigger) -> Result<Self, Self::Error> {
        let trigger_type = match raw.type_ {
            0 => GrammarTriggerType::Token,
            1 => GrammarTriggerType::Word,
            2 => GrammarTriggerType::Pattern,
            3 => GrammarTriggerType::PatternFull,
            _ => return Err(ApplyChatTemplateError::InvalidGrammarTriggerType),
        };
        let token = if trigger_type == GrammarTriggerType::Token {
            Some(LlamaToken(raw.token))
        } else {
            None
        };

        Ok(Self {
            trigger_type,
            value: raw.value,
            token,
        })
    }
}

/// Parse a JSON string returned by `llama_rs_apply_chat_template_*` into a [`ChatTemplateResult`].
///
/// Frees the C-allocated JSON pointer regardless of outcome.
///
/// # Safety
///
/// `out_json` must either be null or a heap-allocated null-terminated string produced by the
/// FFI call associated with `ffi_return_code`. The pointer is freed via `llama_rs_string_free`.
///
/// # Errors
///
/// Returns [`ApplyChatTemplateError`] on FFI failure, null pointer, non-UTF-8 payload,
/// JSON parse failure, or unknown grammar trigger type.
pub unsafe fn parse_chat_template_json_result(
    ffi_return_code: llama_cpp_bindings_sys::llama_rs_status,
    out_json: *mut c_char,
    parse_tool_calls: bool,
) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
    let result = (|| {
        if !status_is_ok(ffi_return_code) {
            return Err(ApplyChatTemplateError::FfiError(status_to_i32(
                ffi_return_code,
            )));
        }
        if out_json.is_null() {
            return Err(ApplyChatTemplateError::NullResult);
        }

        let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
        let json_str = String::from_utf8(bytes)?;
        let raw: RawTemplateResult = serde_json::from_str(&json_str)?;

        let grammar_triggers = raw
            .grammar_triggers
            .into_iter()
            .map(GrammarTrigger::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatTemplateResult {
            prompt: raw.prompt,
            grammar: raw.grammar,
            grammar_lazy: raw.grammar_lazy,
            grammar_triggers,
            preserved_tokens: raw.preserved_tokens,
            additional_stops: raw.additional_stops,
            chat_format: raw.chat_format,
            parser: raw.parser,
            generation_prompt: raw.generation_prompt,
            supports_thinking: raw.supports_thinking,
            parse_tool_calls,
        })
    })();

    if !out_json.is_null() {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };
    }

    result
}

impl ChatTemplateResult {
    /// Parse a generated response into an OpenAI-compatible message JSON string.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result is null.
    pub fn parse_response_oaicompat(
        &self,
        text: &str,
        is_partial: bool,
    ) -> Result<String, ChatParseError> {
        let text_cstr = CString::new(text)?;
        let parser_cstr = self.parser.as_deref().map(CString::new).transpose()?;
        let generation_prompt_cstr = if self.generation_prompt.is_empty() {
            None
        } else {
            Some(CString::new(self.generation_prompt.as_str())?)
        };
        let mut out_json: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_to_oaicompat(
                text_cstr.as_ptr(),
                is_partial,
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                generation_prompt_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                &raw mut out_json,
            )
        };

        let result = (|| {
            if !status_is_ok(rc) {
                return Err(ChatParseError::FfiError(status_to_i32(rc)));
            }
            if out_json.is_null() {
                return Err(ChatParseError::NullResult);
            }
            let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
            Ok(String::from_utf8(bytes)?)
        })();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };

        result
    }

    /// Initialize a streaming parser for OpenAI-compatible chat deltas.
    ///
    /// # Errors
    /// Returns an error if the parser state cannot be initialized.
    pub fn streaming_state_oaicompat(&self) -> Result<ChatParseStateOaicompat, ChatParseError> {
        let parser_cstr = self.parser.as_deref().map(CString::new).transpose()?;
        let generation_prompt_cstr = if self.generation_prompt.is_empty() {
            None
        } else {
            Some(CString::new(self.generation_prompt.as_str())?)
        };
        let state = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_init_oaicompat(
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                generation_prompt_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            )
        };
        let state = NonNull::new(state).ok_or(ChatParseError::NullResult)?;

        Ok(ChatParseStateOaicompat { state })
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatTemplateResult, parse_chat_template_json_result};
    use std::ffi::CString;

    fn json_to_cstr(json: &str) -> *mut std::ffi::c_char {
        CString::new(json).unwrap().into_raw()
    }

    #[test]
    fn parse_template_json_minimal_payload() {
        let json = json_to_cstr(
            r#"{"prompt":"hi","chat_format":0,"supports_thinking":false,"grammar_lazy":false,"grammar_triggers":[],"preserved_tokens":[],"additional_stops":[]}"#,
        );
        let result = unsafe {
            parse_chat_template_json_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, json, false)
        };

        let parsed = result.unwrap();
        assert_eq!(parsed.prompt, "hi");
        assert_eq!(parsed.grammar, None);
        assert_eq!(parsed.parser, None);
        assert!(parsed.generation_prompt.is_empty());
        assert!(!parsed.supports_thinking);
        assert!(!parsed.grammar_lazy);
        assert!(!parsed.parse_tool_calls);
    }

    #[test]
    fn parse_template_json_full_payload() {
        let json = json_to_cstr(
            r#"{"prompt":"p","chat_format":7,"supports_thinking":true,"grammar_lazy":true,"grammar":"g","parser":"pg","generation_prompt":"<|a|>","grammar_triggers":[{"type":0,"value":"<tool>","token":42},{"type":1,"value":"function","token":0},{"type":2,"value":"\\{.*\\}","token":0},{"type":3,"value":"^tool$","token":0}],"preserved_tokens":["x","y"],"additional_stops":["</s>"]}"#,
        );
        let parsed = unsafe {
            parse_chat_template_json_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, json, true)
        }
        .unwrap();

        assert_eq!(parsed.prompt, "p");
        assert_eq!(parsed.chat_format, 7);
        assert!(parsed.supports_thinking);
        assert!(parsed.grammar_lazy);
        assert_eq!(parsed.grammar.as_deref(), Some("g"));
        assert_eq!(parsed.parser.as_deref(), Some("pg"));
        assert_eq!(parsed.generation_prompt, "<|a|>");
        assert_eq!(parsed.grammar_triggers.len(), 4);
        assert_eq!(
            parsed.grammar_triggers[0].trigger_type,
            crate::model::grammar_trigger::GrammarTriggerType::Token
        );
        assert_eq!(
            parsed.grammar_triggers[0].token,
            Some(crate::token::LlamaToken(42))
        );
        assert_eq!(parsed.grammar_triggers[1].token, None);
        assert_eq!(
            parsed.preserved_tokens,
            vec!["x".to_string(), "y".to_string()]
        );
        assert_eq!(parsed.additional_stops, vec!["</s>".to_string()]);
        assert!(parsed.parse_tool_calls);
    }

    #[test]
    fn parse_template_json_unknown_grammar_trigger_type_returns_error() {
        let json = json_to_cstr(
            r#"{"prompt":"p","chat_format":0,"supports_thinking":false,"grammar_lazy":false,"grammar_triggers":[{"type":99,"value":"x","token":0}],"preserved_tokens":[],"additional_stops":[]}"#,
        );
        let result = unsafe {
            parse_chat_template_json_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, json, false)
        };

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
    }

    #[test]
    fn parse_template_json_malformed_returns_json_parse_error() {
        let json = json_to_cstr("not json");
        let result = unsafe {
            parse_chat_template_json_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, json, false)
        };

        assert!(result.unwrap_err().to_string().contains("json parse error"));
    }

    #[test]
    fn parse_template_json_missing_prompt_returns_json_parse_error() {
        let json = json_to_cstr(
            r#"{"chat_format":0,"supports_thinking":false,"grammar_lazy":false,"grammar_triggers":[],"preserved_tokens":[],"additional_stops":[]}"#,
        );
        let result = unsafe {
            parse_chat_template_json_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, json, false)
        };

        assert!(result.unwrap_err().to_string().contains("json parse error"));
    }

    #[test]
    fn parse_template_json_ffi_error_returns_ffi_error() {
        let result = unsafe {
            parse_chat_template_json_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT,
                std::ptr::null_mut(),
                false,
            )
        };

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn parse_template_json_null_pointer_with_ok_status_returns_null_result() {
        let result = unsafe {
            parse_chat_template_json_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                std::ptr::null_mut(),
                false,
            )
        };

        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn parse_response_content_only_format() {
        let json_string = ChatTemplateResult::default()
            .parse_response_oaicompat("Hello, world!", false)
            .unwrap();
        let json_value: serde_json::Value = serde_json::from_str(&json_string).unwrap();
        assert_eq!(json_value["role"], "assistant");
        assert_eq!(json_value["content"], "Hello, world!");
    }

    #[test]
    fn parse_response_null_byte_returns_error() {
        let result = ChatTemplateResult::default().parse_response_oaicompat("hello\0world", false);
        assert!(result.is_err());
    }

    #[test]
    fn parse_response_with_null_byte_parser_returns_error() {
        let template_result = ChatTemplateResult {
            parser: Some("null\0byte".to_string()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.parse_response_oaicompat("hello", false);

        assert!(result.is_err());
    }

    #[test]
    fn parse_response_with_valid_parser() {
        let template_result = ChatTemplateResult {
            parser: Some(String::new()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.parse_response_oaicompat("hello", false);

        assert!(result.is_ok());
    }

    #[test]
    fn parse_response_with_generation_prompt_succeeds() {
        let template_result = ChatTemplateResult {
            generation_prompt: "<|assistant|>".to_string(),
            ..ChatTemplateResult::default()
        };

        let result = template_result.parse_response_oaicompat("hello", false);

        assert!(result.is_ok());
    }

    #[test]
    fn parse_response_with_null_byte_generation_prompt_returns_error() {
        let template_result = ChatTemplateResult {
            generation_prompt: "null\0byte".to_string(),
            ..ChatTemplateResult::default()
        };

        let result = template_result.parse_response_oaicompat("hello", false);

        assert!(result.is_err());
    }

    #[test]
    fn streaming_state_returns_valid_state() {
        let template_result = ChatTemplateResult::default();
        let state = template_result.streaming_state_oaicompat();
        assert!(state.is_ok());
    }

    #[test]
    fn streaming_state_with_null_byte_parser_returns_error() {
        let template_result = ChatTemplateResult {
            parser: Some("null\0byte".to_string()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

        assert!(result.is_err());
    }

    #[test]
    fn streaming_state_with_valid_parser() {
        let template_result = ChatTemplateResult {
            parser: Some(String::new()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

        assert!(result.is_ok());
    }

    #[test]
    fn streaming_state_with_generation_prompt_succeeds() {
        let template_result = ChatTemplateResult {
            generation_prompt: "<|assistant|>".to_string(),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

        assert!(result.is_ok());
    }

    #[test]
    fn streaming_state_with_null_byte_generation_prompt_returns_error() {
        let template_result = ChatTemplateResult {
            generation_prompt: "null\0byte".to_string(),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

        assert!(result.is_err());
    }
}
