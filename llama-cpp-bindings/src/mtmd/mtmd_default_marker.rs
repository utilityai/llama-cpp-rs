use std::ffi::CStr;
use std::os::raw::c_char;

use crate::mtmd::mtmd_default_marker_error::MtmdDefaultMarkerError;

unsafe fn marker_bytes_to_str(
    c_str: *const c_char,
) -> Result<&'static str, MtmdDefaultMarkerError> {
    Ok(unsafe { CStr::from_ptr(c_str) }.to_str()?)
}

/// # Errors
///
/// Returns [`MtmdDefaultMarkerError::NotUtf8`] if llama.cpp's `mtmd_default_marker`
/// returns bytes that are not valid UTF-8. The marker is a fixed ASCII constant;
/// surfacing the error keeps the failure explicit rather than papering over it with
/// a substituted literal.
pub fn mtmd_default_marker() -> Result<&'static str, MtmdDefaultMarkerError> {
    unsafe { marker_bytes_to_str(llama_cpp_bindings_sys::mtmd_default_marker()) }
}

#[cfg(test)]
mod tests {
    use std::os::raw::c_char;

    use super::marker_bytes_to_str;
    use super::mtmd_default_marker;
    use crate::mtmd::mtmd_default_marker_error::MtmdDefaultMarkerError;

    #[test]
    fn returns_non_empty_marker() {
        let marker = mtmd_default_marker().expect("marker must be valid UTF-8");
        assert!(!marker.is_empty());
    }

    #[test]
    fn non_utf8_marker_bytes_return_not_utf8_error() {
        let invalid: [u8; 3] = [0xFF, 0xFE, 0x00];
        let result = unsafe { marker_bytes_to_str(invalid.as_ptr().cast::<c_char>()) };

        assert!(matches!(result, Err(MtmdDefaultMarkerError::NotUtf8(_))));
    }
}
