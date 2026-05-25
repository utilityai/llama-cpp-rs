#[must_use]
pub const fn status_is_ok(status: llama_cpp_bindings_sys::llama_rs_status) -> bool {
    status == llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::status_is_ok;

    #[test]
    fn ok_status() {
        assert!(status_is_ok(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK));
    }

    #[test]
    fn error_status() {
        assert!(!status_is_ok(1));
        assert!(!status_is_ok(-1));
    }
}
