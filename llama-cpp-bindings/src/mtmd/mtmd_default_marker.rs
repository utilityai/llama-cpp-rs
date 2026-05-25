use std::ffi::CStr;

#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = llama_cpp_bindings_sys::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

#[cfg(test)]
mod tests {
    use super::mtmd_default_marker;

    #[test]
    fn returns_non_empty_string() {
        let marker = mtmd_default_marker();
        assert!(!marker.is_empty());
    }
}
