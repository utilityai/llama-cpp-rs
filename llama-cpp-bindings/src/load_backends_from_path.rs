use std::ffi::CString;
use std::path::Path;

use crate::load_backends_error::LoadBackendsError;

/// Load GGML backend modules from the given directory.
///
/// Call this before [`crate::llama_backend::LlamaBackend::init`] to enable runtime hardware
/// selection (Vulkan, CPU-AVX512, CPU-AVX2, etc.) when built with the `dynamic-backends` feature.
///
/// # Errors
///
/// Returns [`LoadBackendsError::PathNotUtf8`] when `path` cannot be converted to UTF-8 and
/// [`LoadBackendsError::PathNullByte`] when the path contains an interior null byte.
pub fn load_backends_from_path(path: &Path) -> Result<(), LoadBackendsError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| LoadBackendsError::PathNotUtf8(path.to_path_buf()))?;
    let path_cstring = CString::new(path_str)?;

    unsafe {
        llama_cpp_bindings_sys::ggml_backend_load_all_from_path(path_cstring.as_ptr());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::load_backends_from_path;
    use crate::load_backends_error::LoadBackendsError;
    use std::path::PathBuf;

    #[test]
    #[cfg(unix)]
    fn load_backends_from_path_returns_path_null_byte_for_embedded_null() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/foo\0bar"));
        let result = load_backends_from_path(&path);

        assert!(matches!(result, Err(LoadBackendsError::PathNullByte(_))));
    }

    #[test]
    #[cfg(unix)]
    fn load_backends_from_path_returns_path_not_utf8_for_invalid_utf8() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/\xff\xfe"));
        let result = load_backends_from_path(&path);

        assert!(matches!(result, Err(LoadBackendsError::PathNotUtf8(_))));
    }
}
