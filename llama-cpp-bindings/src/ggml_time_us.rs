/// Get the time in microseconds according to ggml.
///
/// ```
/// # use std::time::Duration;
/// # use llama_cpp_bindings::llama_backend::LlamaBackend;
/// let backend = LlamaBackend::init().unwrap();
/// use llama_cpp_bindings::ggml_time_us;
///
/// let start = ggml_time_us();
///
/// std::thread::sleep(Duration::from_micros(10));
///
/// let end = ggml_time_us();
///
/// let elapsed = end - start;
///
/// assert!(elapsed >= 10)
#[must_use]
pub fn ggml_time_us() -> i64 {
    unsafe { llama_cpp_bindings_sys::ggml_time_us() }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::ggml_time_us;
    use crate::llama_backend::LlamaBackend;

    #[test]
    #[serial]
    fn returns_positive_value() {
        let _backend = LlamaBackend::init().unwrap();
        let time_microseconds = ggml_time_us();

        assert!(time_microseconds > 0);
    }
}
