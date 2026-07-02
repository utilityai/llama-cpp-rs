use std::ffi::{CStr, CString, c_char};

use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;
use crate::ffi_error_reader::read_and_free_cpp_error;

/// # Safety
///
/// On `LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK` the function reads and frees `out` as a
/// null-terminated C string allocated by the wrapper, so `out` must be a valid such
/// pointer for that status. On error statuses it reads and frees `error_ptr` via
/// [`read_and_free_cpp_error`], which tolerates a null pointer.
unsafe fn json_schema_to_grammar_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar_status,
    out: *mut c_char,
    error_ptr: *mut c_char,
) -> Result<String, JsonSchemaToGrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK => {
            let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };
            Ok(String::from_utf8(grammar_bytes)?)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(JsonSchemaToGrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(JsonSchemaToGrammarError::InvalidSchema { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(JsonSchemaToGrammarError::Reported { message })
        }
        other => Err(JsonSchemaToGrammarError::UnrecognizedStatusCode { code: other }),
    }
}

/// # Errors
///
/// Returns [`JsonSchemaToGrammarError`] if the schema string contains a NUL byte,
/// the wrapper reports any non-OK status, or the returned grammar is not valid UTF-8.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String, JsonSchemaToGrammarError> {
    let schema_cstr = CString::new(schema_json)?;
    let mut out: *mut c_char = std::ptr::null_mut();
    let mut error_ptr: *mut c_char = std::ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema_cstr.as_ptr(),
            false,
            &raw mut out,
            &raw mut error_ptr,
        )
    };

    unsafe { json_schema_to_grammar_status_to_result(status, out, error_ptr) }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;

    use super::json_schema_to_grammar;
    use super::json_schema_to_grammar_status_to_result;
    use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;

    unsafe extern "C" {
        fn strdup(source: *const c_char) -> *mut c_char;
    }

    #[test]
    fn simple_object() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let grammar = json_schema_to_grammar(schema).expect("schema converts to grammar");

        assert!(!grammar.is_empty());
    }

    #[test]
    fn null_byte_returns_schema_contains_nul_byte_error() {
        use std::ffi::CString;

        let schema = "{\x00}";
        let err = json_schema_to_grammar(schema).unwrap_err();
        let representative = JsonSchemaToGrammarError::SchemaContainsNulByte(
            CString::new(b"a\0b".to_vec()).unwrap_err(),
        );

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }

    #[test]
    fn simple_string() {
        let schema = r#"{"type": "string"}"#;
        let grammar = json_schema_to_grammar(schema).expect("schema converts to grammar");

        assert!(!grammar.is_empty());
    }

    #[test]
    fn invalid_json_returns_reported() {
        let schema = "not valid json at all";
        let err = json_schema_to_grammar(schema).unwrap_err();
        let representative = JsonSchemaToGrammarError::Reported {
            message: String::new(),
        };

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }

    #[test]
    fn unresolved_ref_returns_invalid_schema() {
        let schema = r##"{"$ref": "#/$defs/Missing"}"##;
        let err = json_schema_to_grammar(schema).unwrap_err();
        let representative = JsonSchemaToGrammarError::InvalidSchema {
            message: String::new(),
        };

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }

    #[test]
    fn invalid_schema_status_returns_invalid_schema() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::InvalidSchema {
                message: "unknown error".to_owned(),
            })
        );
    }

    #[test]
    fn exception_status_returns_reported() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_THREW_CXX_EXCEPTION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::Reported {
                message: "unknown error".to_owned(),
            })
        );
    }

    #[test]
    fn allocation_failed_status_returns_not_enough_memory() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(JsonSchemaToGrammarError::NotEnoughMemory));
    }

    #[test]
    fn ok_status_with_non_utf8_grammar_returns_grammar_not_utf8() {
        let invalid_utf8_grammar: [u8; 2] = [0xFF, 0];
        let out = unsafe { strdup(invalid_utf8_grammar.as_ptr().cast::<c_char>()) };
        assert!(!out.is_null(), "strdup must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                std::ptr::null_mut(),
            )
        };
        let representative =
            JsonSchemaToGrammarError::GrammarNotUtf8(String::from_utf8(vec![0xFF]).unwrap_err());

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&representative),
        );
    }

    #[test]
    fn ok_status_with_valid_utf8_grammar_returns_grammar_string() {
        let grammar_text: &[u8; 14] = b"root ::= \"x\"\0\0";
        let out = unsafe { strdup(grammar_text.as_ptr().cast::<c_char>()) };
        assert!(!out.is_null(), "strdup must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok("root ::= \"x\"".to_owned()));
    }

    #[test]
    fn unrecognized_status_returns_unrecognized_status_error() {
        assert_eq!(
            unsafe {
                json_schema_to_grammar_status_to_result(
                    llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar_status::MAX,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            Err(JsonSchemaToGrammarError::UnrecognizedStatusCode {
                code: llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar_status::MAX
            }),
        );
    }
}
