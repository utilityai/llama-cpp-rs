#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
}

#[cfg(test)]
mod tests {
    use super::mmap_supported;

    #[test]
    fn returns_bool_without_panic() {
        let _supported: bool = mmap_supported();
    }
}
