#[must_use]
pub fn llama_time_us() -> i64 {
    unsafe { llama_cpp_bindings_sys::llama_time_us() }
}

#[cfg(test)]
mod tests {
    use super::llama_time_us;

    #[test]
    fn returns_positive_value() {
        let time_microseconds = llama_time_us();

        assert!(time_microseconds > 0);
    }
}
