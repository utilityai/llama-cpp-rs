use std::ffi::{CStr, CString, c_char};
use std::ptr::{self, NonNull};
use std::slice;

use crate::model::grammar_trigger::{GrammarTrigger, GrammarTriggerType};
use crate::openai::ChatParseStateOaicompat;
use crate::token::LlamaToken;
use crate::{ApplyChatTemplateError, ChatParseError, status_is_ok, status_to_i32};

const fn check_chat_parse_status(
    rc: llama_cpp_bindings_sys::llama_rs_status,
) -> Result<(), ChatParseError> {
    if !status_is_ok(rc) {
        return Err(ChatParseError::FfiError(status_to_i32(rc)));
    }

    Ok(())
}

const fn check_chat_parse_not_null(json_ptr: *const c_char) -> Result<(), ChatParseError> {
    if json_ptr.is_null() {
        return Err(ChatParseError::NullResult);
    }

    Ok(())
}

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
    /// Whether the model supports thinking/reasoning blocks.
    pub supports_thinking: bool,
    /// Whether tool calls should be parsed from the response.
    pub parse_tool_calls: bool,
}

#[must_use]
pub const fn new_empty_chat_template_raw_result()
-> llama_cpp_bindings_sys::llama_rs_chat_template_result {
    llama_cpp_bindings_sys::llama_rs_chat_template_result {
        prompt: ptr::null_mut(),
        grammar: ptr::null_mut(),
        parser: ptr::null_mut(),
        chat_format: 0,
        supports_thinking: false,
        grammar_lazy: false,
        grammar_triggers: ptr::null_mut(),
        grammar_triggers_count: 0,
        preserved_tokens: ptr::null_mut(),
        preserved_tokens_count: 0,
        additional_stops: ptr::null_mut(),
        additional_stops_count: 0,
    }
}

/// # Safety
///
/// `raw_cstr_array` must point to `count` valid, null-terminated C strings.
unsafe fn parse_raw_cstr_array(
    raw_cstr_array: *const *mut c_char,
    count: usize,
) -> Result<Vec<String>, ApplyChatTemplateError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if raw_cstr_array.is_null() {
        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
    }

    let raw_entries = unsafe { slice::from_raw_parts(raw_cstr_array, count) };
    let mut parsed = Vec::with_capacity(raw_entries.len());

    for entry in raw_entries {
        if entry.is_null() {
            return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
        }
        let bytes = unsafe { CStr::from_ptr(*entry) }.to_bytes().to_vec();
        parsed.push(String::from_utf8(bytes)?);
    }

    Ok(parsed)
}

/// # Safety
///
/// `raw_triggers` must point to `count` valid `llama_rs_grammar_trigger` structs.
unsafe fn parse_raw_grammar_triggers(
    raw_triggers: *const llama_cpp_bindings_sys::llama_rs_grammar_trigger,
    count: usize,
) -> Result<Vec<GrammarTrigger>, ApplyChatTemplateError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if raw_triggers.is_null() {
        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
    }

    let triggers = unsafe { slice::from_raw_parts(raw_triggers, count) };
    let mut parsed = Vec::with_capacity(triggers.len());

    for trigger in triggers {
        let trigger_type = match trigger.type_ {
            0 => GrammarTriggerType::Token,
            1 => GrammarTriggerType::Word,
            2 => GrammarTriggerType::Pattern,
            3 => GrammarTriggerType::PatternFull,
            _ => return Err(ApplyChatTemplateError::InvalidGrammarTriggerType),
        };
        let value = if trigger.value.is_null() {
            return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
        } else {
            let bytes = unsafe { CStr::from_ptr(trigger.value) }.to_bytes().to_vec();
            String::from_utf8(bytes)?
        };
        let token = if trigger_type == GrammarTriggerType::Token {
            Some(LlamaToken(trigger.token))
        } else {
            None
        };
        parsed.push(GrammarTrigger {
            trigger_type,
            value,
            token,
        });
    }

    Ok(parsed)
}

/// # Safety
///
/// `raw_result` must point to a valid, initialized `llama_rs_chat_template_result`.
///
/// # Errors
/// Returns `ApplyChatTemplateError` if the FFI call failed or the result could not be parsed.
pub unsafe fn parse_chat_template_raw_result(
    ffi_return_code: llama_cpp_bindings_sys::llama_rs_status,
    raw_result: *mut llama_cpp_bindings_sys::llama_rs_chat_template_result,
    parse_tool_calls: bool,
) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
    let result = (|| {
        if !status_is_ok(ffi_return_code) {
            return Err(ApplyChatTemplateError::FfiError(status_to_i32(
                ffi_return_code,
            )));
        }

        let raw = unsafe { &*raw_result };

        if raw.prompt.is_null() {
            return Err(ApplyChatTemplateError::NullResult);
        }

        let prompt_bytes = unsafe { CStr::from_ptr(raw.prompt) }.to_bytes().to_vec();
        let prompt = String::from_utf8(prompt_bytes)?;

        let grammar = if raw.grammar.is_null() {
            None
        } else {
            let grammar_bytes = unsafe { CStr::from_ptr(raw.grammar) }.to_bytes().to_vec();
            Some(String::from_utf8(grammar_bytes)?)
        };

        let parser = if raw.parser.is_null() {
            None
        } else {
            let parser_bytes = unsafe { CStr::from_ptr(raw.parser) }.to_bytes().to_vec();
            Some(String::from_utf8(parser_bytes)?)
        };

        let grammar_triggers = unsafe {
            parse_raw_grammar_triggers(raw.grammar_triggers, raw.grammar_triggers_count)
        }?;

        let preserved_tokens =
            unsafe { parse_raw_cstr_array(raw.preserved_tokens, raw.preserved_tokens_count) }?;

        let additional_stops =
            unsafe { parse_raw_cstr_array(raw.additional_stops, raw.additional_stops_count) }?;

        Ok(ChatTemplateResult {
            prompt,
            grammar,
            grammar_lazy: raw.grammar_lazy,
            grammar_triggers,
            preserved_tokens,
            additional_stops,
            chat_format: raw.chat_format,
            parser,
            supports_thinking: raw.supports_thinking,
            parse_tool_calls,
        })
    })();

    unsafe { llama_cpp_bindings_sys::llama_rs_chat_template_result_free(raw_result) };

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
                &raw mut out_json,
            )
        };

        let result = (|| {
            check_chat_parse_status(rc)?;
            check_chat_parse_not_null(out_json)?;
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
        let state = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_init_oaicompat(
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
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
    use std::ffi::{CString, c_char};
    use std::ptr;

    use super::{
        ChatTemplateResult, new_empty_chat_template_raw_result, parse_chat_template_raw_result,
        parse_raw_cstr_array, parse_raw_grammar_triggers,
    };
    use crate::model::grammar_trigger::GrammarTriggerType;
    use crate::token::LlamaToken;

    fn heap_cstring(value: &str) -> *mut c_char {
        CString::new(value).unwrap().into_raw()
    }

    #[test]
    fn parse_cstr_array_zero_count_returns_empty() {
        let result = unsafe { parse_raw_cstr_array(ptr::null(), 0) };
        assert_eq!(result.unwrap(), Vec::<String>::new());
    }

    #[test]
    fn parse_cstr_array_null_with_nonzero_count_returns_error() {
        let result = unsafe { parse_raw_cstr_array(ptr::null(), 1) };
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
    }

    #[test]
    fn parse_cstr_array_valid_single_string() {
        let raw_string = heap_cstring("hello");
        let array = [raw_string];
        let result = unsafe { parse_raw_cstr_array(array.as_ptr(), 1) };
        assert_eq!(result.unwrap(), vec!["hello".to_string()]);
        unsafe { drop(CString::from_raw(array[0])) };
    }

    #[test]
    fn parse_cstr_array_null_entry_returns_error() {
        let raw_string = heap_cstring("valid");
        let array: [*mut c_char; 2] = [raw_string, ptr::null_mut()];
        let result = unsafe { parse_raw_cstr_array(array.as_ptr(), 2) };
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
        unsafe { drop(CString::from_raw(array[0])) };
    }

    #[test]
    fn parse_triggers_zero_count_returns_empty() {
        let result = unsafe { parse_raw_grammar_triggers(ptr::null(), 0) };
        assert_eq!(result.unwrap(), Vec::new());
    }

    #[test]
    fn parse_triggers_null_with_nonzero_count_returns_error() {
        let result = unsafe { parse_raw_grammar_triggers(ptr::null(), 1) };
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
    }

    #[test]
    fn parse_triggers_token_type_has_token() {
        let value_ptr = heap_cstring("<tool>");
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 0,
            value: value_ptr,
            token: 42,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        let parsed = result.unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].trigger_type, GrammarTriggerType::Token);
        assert_eq!(parsed[0].value, "<tool>");
        assert_eq!(parsed[0].token, Some(LlamaToken(42)));
        unsafe { drop(CString::from_raw(value_ptr)) };
    }

    #[test]
    fn parse_triggers_word_type_has_no_token() {
        let value_ptr = heap_cstring("function");
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 1,
            value: value_ptr,
            token: 99,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        let parsed = result.unwrap();
        assert_eq!(parsed[0].trigger_type, GrammarTriggerType::Word);
        assert_eq!(parsed[0].token, None);
        unsafe { drop(CString::from_raw(value_ptr)) };
    }

    #[test]
    fn parse_triggers_pattern_type() {
        let value_ptr = heap_cstring("\\{.*\\}");
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 2,
            value: value_ptr,
            token: 0,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        assert_eq!(result.unwrap()[0].trigger_type, GrammarTriggerType::Pattern);
        unsafe { drop(CString::from_raw(value_ptr)) };
    }

    #[test]
    fn parse_triggers_pattern_full_type() {
        let value_ptr = heap_cstring("^tool$");
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 3,
            value: value_ptr,
            token: 0,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        assert_eq!(
            result.unwrap()[0].trigger_type,
            GrammarTriggerType::PatternFull
        );
        unsafe { drop(CString::from_raw(value_ptr)) };
    }

    #[test]
    fn parse_triggers_invalid_type_returns_error() {
        let value_ptr = heap_cstring("x");
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 4,
            value: value_ptr,
            token: 0,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
        unsafe { drop(CString::from_raw(value_ptr)) };
    }

    #[test]
    fn parse_triggers_null_value_returns_error() {
        let trigger = llama_cpp_bindings_sys::llama_rs_grammar_trigger {
            type_: 1,
            value: ptr::null_mut(),
            token: 0,
        };
        let result = unsafe { parse_raw_grammar_triggers(&raw const trigger, 1) };
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
    }

    #[test]
    fn parse_raw_result_error_status_returns_ffi_error() {
        let mut raw = new_empty_chat_template_raw_result();
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT,
                &raw mut raw,
                false,
            )
        };
        assert!(result.unwrap_err().to_string().contains("ffi error -1"));
    }

    #[test]
    fn parse_raw_result_null_prompt_returns_null_result() {
        let mut raw = new_empty_chat_template_raw_result();
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };
        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn parse_raw_result_minimal_prompt() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("Hello");
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };
        let parsed = result.unwrap();
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.grammar, None);
        assert_eq!(parsed.parser, None);
        assert!(!parsed.supports_thinking);
        assert!(!parsed.grammar_lazy);
        assert!(!parsed.parse_tool_calls);
    }

    #[test]
    fn parse_raw_result_supports_thinking_true() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("test");
        raw.supports_thinking = true;
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };
        assert!(result.unwrap().supports_thinking);
    }

    #[test]
    fn parse_raw_result_with_grammar_and_parser() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("prompt");
        raw.grammar = heap_cstring("root ::= .*");
        raw.parser = heap_cstring("peg_data");
        raw.grammar_lazy = true;
        raw.chat_format = 2;
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                true,
            )
        };
        let parsed = result.unwrap();
        assert_eq!(parsed.grammar.as_deref(), Some("root ::= .*"));
        assert_eq!(parsed.parser.as_deref(), Some("peg_data"));
        assert!(parsed.grammar_lazy);
        assert_eq!(parsed.chat_format, 2);
        assert!(parsed.parse_tool_calls);
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
    fn parse_raw_result_invalid_triggers_propagates_error() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("prompt");
        raw.grammar_triggers = ptr::null_mut();
        raw.grammar_triggers_count = 1;
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid grammar trigger data")
        );
    }

    #[test]
    fn check_chat_parse_status_ok() {
        let result = super::check_chat_parse_status(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK);

        assert!(result.is_ok());
    }

    #[test]
    fn check_chat_parse_status_error() {
        let result = super::check_chat_parse_status(
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT,
        );

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn check_chat_parse_not_null_ok() {
        let cstr = CString::new("test").unwrap();
        let result = super::check_chat_parse_not_null(cstr.as_ptr());

        assert!(result.is_ok());
    }

    #[test]
    fn check_chat_parse_not_null_error() {
        let result = super::check_chat_parse_not_null(ptr::null());

        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn streaming_state_returns_valid_state() {
        let template_result = ChatTemplateResult::default();
        let state = template_result.streaming_state_oaicompat();
        assert!(state.is_ok());
    }

    #[test]
    fn parse_raw_result_null_preserved_token_propagates_error() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("test");
        raw.preserved_tokens_count = 1;
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };

        assert!(result.is_err());
    }

    #[test]
    fn parse_raw_result_null_additional_stop_propagates_error() {
        let mut raw = new_empty_chat_template_raw_result();
        raw.prompt = heap_cstring("test");
        raw.additional_stops_count = 1;
        let result = unsafe {
            parse_chat_template_raw_result(
                llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK,
                &raw mut raw,
                false,
            )
        };

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
    fn streaming_state_with_null_byte_parser_returns_error() {
        let template_result = ChatTemplateResult {
            parser: Some("null\0byte".to_string()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

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
    fn streaming_state_with_valid_parser() {
        let template_result = ChatTemplateResult {
            parser: Some(String::new()),
            ..ChatTemplateResult::default()
        };

        let result = template_result.streaming_state_oaicompat();

        assert!(result.is_ok());
    }
}
