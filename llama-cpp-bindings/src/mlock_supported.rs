#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
}

#[cfg(test)]
mod tests {
    use super::mlock_supported;

    #[test]
    fn returns_bool_without_panic() {
        let _supported: bool = mlock_supported();
    }
}
