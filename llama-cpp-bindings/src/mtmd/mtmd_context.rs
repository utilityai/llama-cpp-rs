use std::ffi::CString;
use std::ffi::c_char;
use std::ptr::NonNull;

use crate::ffi_error_reader::read_and_free_cpp_error;
use crate::model::LlamaModel;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_context_params::MtmdContextParams;
use super::mtmd_encode_error::MtmdEncodeError;
use super::mtmd_init_error::MtmdInitError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;
use super::mtmd_input_text::MtmdInputText;
use super::mtmd_tokenize_error::MtmdTokenizeError;

fn map_tokenize_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdTokenizeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT => {
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_IMAGE_PREPROCESSING_ERROR => {
            Err(MtmdTokenizeError::MediaPreprocessingFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdTokenizeError::UnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdTokenizeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(out_error) };
            Err(MtmdTokenizeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO => unreachable!("llama_rs_mtmd_tokenize NULL_BITMAPS_ARG: Rust always passes a non-null bitmaps pointer when count > 0"),
        other => unreachable!("llama_rs_mtmd_tokenize returned unrecognized status: {other}"),
    }
}

fn map_encode_chunk_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk_status,
    vendored_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdEncodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_RETURNED_NONZERO_CODE => {
            Err(MtmdEncodeError::EncodingFailed {
                code: vendored_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdEncodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(out_error) };
            Err(MtmdEncodeError::Reported { message })
        }
        other => unreachable!("llama_rs_mtmd_encode_chunk returned unrecognized status: {other}"),
    }
}

/// Safe wrapper around `mtmd_context`.
///
/// This represents an initialized multimodal context that can process
/// text, images, and audio through llama.cpp's multimodal interface.
#[derive(Debug)]
pub struct MtmdContext {
    /// Raw pointer to the underlying `mtmd_context`.
    pub context: NonNull<llama_cpp_bindings_sys::mtmd_context>,
}

unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Initialize MTMD context from a multimodal projection file.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdInitError`] variant matching the wrapper's status code.
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = llama_cpp_bindings_sys::mtmd_context_params::from(params);

        let mut out_ctx: *mut llama_cpp_bindings_sys::mtmd_context = std::ptr::null_mut();
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };

        match status {
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK => {
                let context = NonNull::new(out_ctx).ok_or_else(|| MtmdInitError::Unloadable {
                    path: std::path::PathBuf::from(mmproj_path),
                })?;
                Ok(Self { context })
            }
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_RETURNED_NULL => {
                Err(MtmdInitError::Unloadable {
                    path: std::path::PathBuf::from(mmproj_path),
                })
            }
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
                Err(MtmdInitError::NotEnoughMemory)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { read_and_free_cpp_error(out_error) };
                Err(MtmdInitError::Reported { message })
            }
            other => {
                unreachable!("llama_rs_mtmd_init_from_file returned unrecognized status: {other}")
            }
        }
    }

    /// Check whether non-causal attention mask is needed before `llama_decode`
    /// for the given input chunk.
    #[must_use]
    pub fn decode_use_non_causal(&self, chunk: &MtmdInputChunk) -> bool {
        unsafe {
            llama_cpp_bindings_sys::mtmd_decode_use_non_causal(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
            )
        }
    }

    /// Check whether the current model uses M-RoPE for `llama_decode`.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check whether the current model supports vision input.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check whether the current model supports audio input.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Get audio sample rate in Hz (e.g., 16000 for Whisper).
    /// Returns None if audio is not supported.
    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate =
            unsafe { llama_cpp_bindings_sys::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// Tokenize input text and bitmaps into chunks.
    ///
    /// The input text must contain media markers (default: `<__media__>`) that will be
    /// replaced with the corresponding bitmap data from the `bitmaps` array.
    /// The number of bitmaps must equal the number of markers in the text.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdTokenizeError`] variant matching the wrapper's status code.
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new()?;
        let text_cstring = CString::new(text.text)?;
        let input_text = llama_cpp_bindings_sys::mtmd_input_text {
            text: text_cstring.as_ptr(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        let bitmap_ptrs: Vec<*const llama_cpp_bindings_sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|bitmap| bitmap.bitmap.as_ptr().cast_const())
            .collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
                &raw mut out_undocumented_return_code,
                &raw mut out_error,
            )
        };

        map_tokenize_status(status, out_undocumented_return_code, out_error)?;
        Ok(chunks)
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdEncodeError`] variant matching the wrapper's status code.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk) -> Result<(), MtmdEncodeError> {
        let mut out_vendored_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
                &raw mut out_vendored_return_code,
                &raw mut out_error,
            )
        };

        map_encode_chunk_status(status, out_vendored_return_code, out_error)
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::map_encode_chunk_status;
    use super::map_tokenize_status;
    use crate::mtmd::mtmd_encode_error::MtmdEncodeError;
    use crate::mtmd::mtmd_tokenize_error::MtmdTokenizeError;

    #[test]
    fn tokenize_status_maps_bitmap_count_mismatch() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
            0,
            std::ptr::null_mut(),
        );

        assert!(matches!(
            result,
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        ));
    }

    #[test]
    fn tokenize_status_maps_media_preprocessing_failed() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_IMAGE_PREPROCESSING_ERROR,
            0,
            std::ptr::null_mut(),
        );

        assert!(matches!(
            result,
            Err(MtmdTokenizeError::MediaPreprocessingFailed)
        ));
    }

    #[test]
    fn tokenize_status_maps_unknown_status_with_value() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_RETURNED_UNDOCUMENTED_NONZERO_CODE,
            42,
            std::ptr::null_mut(),
        );

        assert!(matches!(
            result,
            Err(MtmdTokenizeError::UnknownStatus { code: 42 })
        ));
    }

    #[test]
    fn tokenize_status_maps_ok_to_unit() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK,
            0,
            std::ptr::null_mut(),
        );

        assert!(matches!(result, Ok(())));
    }

    #[test]
    fn encode_chunk_status_maps_ok_to_unit() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK,
            0,
            std::ptr::null_mut(),
        );

        assert!(matches!(result, Ok(())));
    }

    #[test]
    fn encode_chunk_status_maps_encoding_failed_with_code() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_RETURNED_NONZERO_CODE,
            5,
            std::ptr::null_mut(),
        );

        assert!(matches!(
            result,
            Err(MtmdEncodeError::EncodingFailed { code: 5 })
        ));
    }
}
