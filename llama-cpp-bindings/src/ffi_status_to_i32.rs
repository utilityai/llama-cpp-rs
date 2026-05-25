#[must_use]
pub const fn status_to_i32(status: llama_cpp_bindings_sys::llama_rs_status) -> i32 {
    status
}

#[cfg(test)]
mod tests {
    use super::status_to_i32;

    #[test]
    fn ok_status_converts_to_zero() {
        let result = status_to_i32(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK);

        assert_eq!(result, 0);
    }

    #[test]
    fn error_status_converts_to_negative() {
        let result = status_to_i32(llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);

        assert_eq!(result, -1);
    }
}
