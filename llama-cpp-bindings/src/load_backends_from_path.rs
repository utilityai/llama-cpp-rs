use std::ffi::CString;
use std::path::Path;

use crate::load_backends_error::LoadBackendsError;

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
        use std::ffi::CString;
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/foo\0bar"));
        let err = load_backends_from_path(&path).unwrap_err();
        let representative =
            LoadBackendsError::PathNullByte(CString::new(b"a\0b".to_vec()).unwrap_err());

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }

    #[test]
    #[cfg(unix)]
    fn load_backends_from_path_returns_path_not_utf8_for_invalid_utf8() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/\xff\xfe"));
        let err = load_backends_from_path(&path).unwrap_err();
        let representative = LoadBackendsError::PathNotUtf8(PathBuf::new());

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }
}
