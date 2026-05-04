use std::ffi::{CStr, CString, c_char};
use std::ptr::{self, NonNull};

use crate::{ChatParseError, status_is_ok, status_to_i32};

const fn check_ffi_status(
    status: llama_cpp_bindings_sys::llama_rs_status,
) -> Result<(), ChatParseError> {
    if status_is_ok(status) {
        Ok(())
    } else {
        Err(ChatParseError::FfiError(status_to_i32(status)))
    }
}

const fn check_json_not_null(json_ptr: *const c_char) -> Result<(), ChatParseError> {
    if json_ptr.is_null() {
        Err(ChatParseError::NullResult)
    } else {
        Ok(())
    }
}

/// Streaming OpenAI-compatible parser state.
#[derive(Debug)]
pub struct ChatParseStateOaicompat {
    /// Raw pointer to the underlying FFI parser state.
    pub state: NonNull<llama_cpp_bindings_sys::llama_rs_chat_parse_state_oaicompat>,
}

impl ChatParseStateOaicompat {
    /// Update the parser with additional text and return OpenAI-compatible deltas as JSON strings.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails, the JSON pointer is null, or the JSON
    /// payload returned by the C wrapper is not a valid array of objects.
    pub fn update(
        &mut self,
        text_added: &str,
        is_partial: bool,
    ) -> Result<Vec<String>, ChatParseError> {
        let text_cstr = CString::new(text_added)?;
        let mut out_json: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_update_oaicompat(
                self.state.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &raw mut out_json,
            )
        };

        let result = (|| {
            check_ffi_status(rc)?;
            check_json_not_null(out_json)?;
            let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
            let json_str = String::from_utf8(bytes)?;
            split_diff_array(&json_str)
        })();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };

        result
    }
}

fn split_diff_array(json_array: &str) -> Result<Vec<String>, ChatParseError> {
    let values: Vec<serde_json::Value> = serde_json::from_str(json_array)?;

    Ok(values.into_iter().map(|value| value.to_string()).collect())
}

impl Drop for ChatParseStateOaicompat {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_free_oaicompat(self.state.as_ptr());
        };
    }
}

#[cfg(test)]
mod tests {
    use super::split_diff_array;
    use crate::model::chat_template_result::ChatTemplateResult;

    fn content_only_template() -> ChatTemplateResult {
        ChatTemplateResult::default()
    }

    #[test]
    fn update_with_simple_text() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let deltas = state.update("Hello", true);
        assert!(deltas.is_ok());
    }

    #[test]
    fn update_null_byte_returns_error() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let result = state.update("hello\0world", true);
        assert!(result.unwrap_err().to_string().contains("nul byte"));
    }

    #[test]
    fn update_finalized_produces_deltas() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let deltas = state.update("Hello world", false).unwrap();

        assert!(!deltas.is_empty());
    }

    #[test]
    fn update_empty_text_returns_empty_diff_list() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let deltas = state.update("", true).unwrap();

        assert!(deltas.is_empty());
    }

    #[test]
    fn check_ffi_status_returns_error_for_invalid() {
        let result =
            super::check_ffi_status(llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn check_ffi_status_returns_ok_for_ok_status() {
        let result = super::check_ffi_status(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK);

        assert!(result.is_ok());
    }

    #[test]
    fn check_json_not_null_returns_error() {
        let result = super::check_json_not_null(std::ptr::null());

        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn check_json_not_null_with_pointer_is_ok() {
        let cstr = std::ffi::CString::new("ok").unwrap();
        let result = super::check_json_not_null(cstr.as_ptr());

        assert!(result.is_ok());
    }

    #[test]
    fn split_diff_array_parses_empty_array() {
        let parts = split_diff_array("[]").unwrap();

        assert!(parts.is_empty());
    }

    #[test]
    fn split_diff_array_parses_single_diff() {
        let parts = split_diff_array(r#"[{"content":"hi"}]"#).unwrap();

        assert_eq!(parts, vec![r#"{"content":"hi"}"#.to_string()]);
    }

    #[test]
    fn split_diff_array_parses_multiple_diffs() {
        let parts = split_diff_array(r#"[{"a":1},{"b":2}]"#).unwrap();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], r#"{"a":1}"#);
        assert_eq!(parts[1], r#"{"b":2}"#);
    }

    #[test]
    fn split_diff_array_rejects_non_json() {
        let result = split_diff_array("not json");

        assert!(result.is_err());
    }
}
