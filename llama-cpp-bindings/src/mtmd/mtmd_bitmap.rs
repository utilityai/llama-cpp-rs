use std::ffi::{CStr, CString, c_char};
use std::ptr::NonNull;
use std::slice;

use super::mtmd_context::MtmdContext;
use super::mtmd_error::MtmdBitmapError;

fn cstr_ptr_to_optional_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        let id = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();

        Some(id)
    }
}

/// Safe wrapper around `mtmd_bitmap`.
///
/// Represents bitmap data for images or audio that can be processed
/// by the multimodal system. For images, data is stored in RGB format.
/// For audio, data is stored as PCM F32 samples.
#[derive(Debug, Clone)]
pub struct MtmdBitmap {
    /// Raw pointer to the underlying `mtmd_bitmap`.
    pub bitmap: NonNull<llama_cpp_bindings_sys::mtmd_bitmap>,
}

unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// Create a bitmap from image data in RGB format.
    ///
    /// # Errors
    ///
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_bindings::mtmd::MtmdBitmap;
    ///
    /// // Create a 2x2 red image
    /// let red_pixel = [255, 0, 0]; // RGB values for red
    /// let image_data = red_pixel.repeat(4); // 2x2 = 4 pixels
    ///
    /// let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data);
    /// assert!(bitmap.is_ok());
    /// ```
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        if nx < 2 || ny < 2 {
            return Err(MtmdBitmapError::ImageDimensionsTooSmall(nx, ny));
        }

        if data.len() != (nx * ny * 3) as usize {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;

        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data in PCM F32 format.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_bindings::mtmd::MtmdBitmap;
    ///
    /// // Create a simple sine wave audio sample
    /// let audio_data: Vec<f32> = (0..100)
    ///     .map(|i| (i as f32 * 0.1).sin())
    ///     .collect();
    ///
    /// let bitmap = MtmdBitmap::from_audio_data(&audio_data);
    /// // Note: This will likely fail without proper MTMD context setup
    /// ```
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe {
            llama_cpp_bindings_sys::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr())
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;

        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// # Errors
    ///
    /// * `CStringError` - Path contains null bytes
    /// * `NullResult` - File could not be loaded or processed
    pub fn from_file(ctx: &MtmdContext, path: &str) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        let bitmap = unsafe {
            llama_cpp_bindings_sys::mtmd_helper_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
            )
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;

        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    pub fn from_buffer(ctx: &MtmdContext, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe {
            llama_cpp_bindings_sys::mtmd_helper_bitmap_init_from_buf(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
            )
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;

        Ok(Self { bitmap })
    }

    /// Get bitmap width in pixels.
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height in pixels.
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as a byte slice.
    ///
    /// For images: RGB format with length `nx * ny * 3`
    /// For audio: PCM F32 format with length `n_samples * 4`
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Check if this bitmap contains audio data (vs image data).
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Get the bitmap's optional ID string.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };

        cstr_ptr_to_optional_string(ptr)
    }

    /// Set the bitmap's ID string.
    ///
    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llama_cpp_bindings::mtmd::MtmdBitmap;
    /// # fn example(bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// bitmap.set_id("image_001")?;
    /// assert_eq!(bitmap.id(), Some("image_001".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id(&self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_bindings_sys::mtmd_bitmap_set_id(self.bitmap.as_ptr(), id_cstr.as_ptr());
        }

        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::MtmdBitmap;
    use super::MtmdBitmapError;

    #[test]
    fn cstr_ptr_to_optional_string_returns_none_for_null() {
        assert!(super::cstr_ptr_to_optional_string(std::ptr::null()).is_none());
    }

    #[test]
    fn cstr_ptr_to_optional_string_returns_some_for_valid() {
        let cstr = std::ffi::CString::new("hello").unwrap();
        let result = super::cstr_ptr_to_optional_string(cstr.as_ptr());

        assert_eq!(result, Some("hello".to_string()));
    }

    #[test]
    fn from_image_data_creates_valid_bitmap() {
        let red_pixel: [u8; 3] = [255, 0, 0];
        let image_data: Vec<u8> = red_pixel.repeat(4);
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        assert_eq!(bitmap.nx(), 2);
        assert_eq!(bitmap.ny(), 2);
        assert!(!bitmap.is_audio());
    }

    #[test]
    fn invalid_data_size_returns_error() {
        let too_short = vec![0u8; 5];
        let result = MtmdBitmap::from_image_data(2, 2, &too_short);
        assert!(result.is_err());
    }

    #[test]
    fn from_image_data_rejects_dimensions_below_minimum() {
        let result_1x1 = MtmdBitmap::from_image_data(1, 1, &[0u8; 3]);
        let result_1x2 = MtmdBitmap::from_image_data(1, 2, &[0u8; 6]);
        let result_2x1 = MtmdBitmap::from_image_data(2, 1, &[0u8; 6]);
        let result_0x0 = MtmdBitmap::from_image_data(0, 0, &[]);

        assert!(matches!(
            result_1x1,
            Err(MtmdBitmapError::ImageDimensionsTooSmall(1, 1))
        ));
        assert!(matches!(
            result_1x2,
            Err(MtmdBitmapError::ImageDimensionsTooSmall(1, 2))
        ));
        assert!(matches!(
            result_2x1,
            Err(MtmdBitmapError::ImageDimensionsTooSmall(2, 1))
        ));
        assert!(matches!(
            result_0x0,
            Err(MtmdBitmapError::ImageDimensionsTooSmall(0, 0))
        ));
    }

    #[test]
    fn set_id_changes_id() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        bitmap.set_id("test_image").unwrap();

        assert_eq!(bitmap.id().as_deref(), Some("test_image"));
    }

    #[test]
    fn from_audio_data_creates_valid_bitmap() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "test fixture casts a small i32 (0..100) to f32 to synthesise a sine wave; \
                      the values are well within f32's exact-representation range"
        )]
        let audio_samples: Vec<f32> = (0..100).map(|index| (index as f32 * 0.1).sin()).collect();
        let bitmap = MtmdBitmap::from_audio_data(&audio_samples).unwrap();

        assert!(bitmap.is_audio());
    }

    #[test]
    fn data_returns_expected_bytes_for_image() {
        let pixel_data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &pixel_data).unwrap();
        let returned_data = bitmap.data();

        assert_eq!(returned_data, &pixel_data);
    }

    #[test]
    fn id_returns_some_by_default() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();

        assert!(bitmap.id().is_some());
    }

    #[test]
    fn id_returns_custom_value_after_set() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        bitmap.set_id("my_image").unwrap();

        assert_eq!(bitmap.id(), Some("my_image".to_string()));
    }

    #[test]
    fn set_id_with_null_byte_returns_error() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        let result = bitmap.set_id("id\0null");

        assert!(result.is_err());
    }
}
