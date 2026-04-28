use std::ffi::{CStr, c_char};

/// Reads a C error string, converts to Rust `String`, and frees the C memory.
///
/// # Safety
///
/// `error_ptr` must be either null or a valid pointer to a null-terminated
/// C string allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_cpp_error(error_ptr: *mut c_char) -> String {
    if error_ptr.is_null() {
        return "unknown error".to_owned();
    }

    let message = unsafe { CStr::from_ptr(error_ptr) }
        .to_string_lossy()
        .into_owned();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(error_ptr) };

    message
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ffi::c_char;

    use super::read_and_free_cpp_error;

    unsafe extern "C" {
        fn strdup(s: *const c_char) -> *mut c_char;
    }

    #[test]
    fn returns_unknown_for_null_pointer() {
        let result = unsafe { read_and_free_cpp_error(std::ptr::null_mut()) };

        assert_eq!(result, "unknown error");
    }

    #[test]
    fn returns_message_for_valid_cstring_pointer() {
        let original = CString::new("expected error message").unwrap();
        let dup_ptr = unsafe { strdup(original.as_ptr()) };
        assert!(!dup_ptr.is_null(), "strdup must allocate a copy");

        let result = unsafe { read_and_free_cpp_error(dup_ptr) };

        assert_eq!(result, "expected error message");
    }
}
