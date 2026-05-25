use std::ffi::{CStr, CString, c_char};

use crate::error::JsonSchemaToGrammarError;
use crate::ffi_error_reader::read_and_free_cpp_error;

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
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(JsonSchemaToGrammarError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_json_schema_to_grammar returned unrecognized status {other}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::json_schema_to_grammar;
    use crate::error::JsonSchemaToGrammarError;

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
}
