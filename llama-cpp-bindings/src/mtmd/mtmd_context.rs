use std::ffi::CString;
use std::ptr::NonNull;

use crate::model::LlamaModel;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_context_params::MtmdContextParams;
use super::mtmd_error::{MtmdEncodeError, MtmdInitError, MtmdTokenizeError};
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;
use super::mtmd_input_text::MtmdInputText;

const fn tokenize_result_to_error(result: i32) -> MtmdTokenizeError {
    match result {
        1 => MtmdTokenizeError::BitmapCountMismatch,
        2 => MtmdTokenizeError::ImagePreprocessingError,
        _ => MtmdTokenizeError::UnknownError(result),
    }
}

const fn check_encode_result(result: i32) -> Result<(), MtmdEncodeError> {
    if result == 0 {
        Ok(())
    } else {
        Err(MtmdEncodeError::EncodeFailure(result))
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
    /// This function will return an error if:
    /// - The path cannot be converted to a C string
    /// - The underlying C function returns null (indicating initialization failure)
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = llama_cpp_bindings_sys::mtmd_context_params::from(params);

        let context = unsafe {
            llama_cpp_bindings_sys::mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
            )
        };

        let context = NonNull::new(context).ok_or(MtmdInitError::NullResult)?;

        Ok(Self { context })
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
    /// * `BitmapCountMismatch` - Number of bitmaps doesn't match number of markers
    /// * `ImagePreprocessingError` - Error occurred during image preprocessing
    /// * `UnknownError` - Other tokenization error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use llama_cpp_bindings::mtmd::*;
    /// # fn example(ctx: &MtmdContext, bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// let text = MtmdInputText {
    ///     text: "Here is an image: <__media__>\nDescribe it.".to_string(),
    ///     add_special: true,
    ///     parse_special: true,
    /// };
    /// let chunks = ctx.tokenize(text, &[bitmap])?;
    /// # Ok(())
    /// # }
    /// ```
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

        let result = unsafe {
            llama_cpp_bindings_sys::mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
            )
        };

        if result == 0 {
            Ok(chunks)
        } else {
            Err(tokenize_result_to_error(result))
        }
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk) -> Result<(), MtmdEncodeError> {
        let result = unsafe {
            llama_cpp_bindings_sys::mtmd_encode_chunk(self.context.as_ptr(), chunk.chunk.as_ptr())
        };

        check_encode_result(result)
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::mtmd::mtmd_bitmap::MtmdBitmap;
    use crate::mtmd::mtmd_input_text::MtmdInputText;
    use crate::test_model;

    #[test]
    #[serial]
    fn init_and_supports_vision() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        assert!(mtmd_ctx.support_vision());
    }

    #[test]
    #[serial]
    fn tokenize_text_with_image() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Describe this image: <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]);
        assert!(chunks.is_ok());
        let chunks = chunks.unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.total_tokens() > 0);
    }

    #[test]
    #[serial]
    fn eval_chunks_with_standard_image() {
        let (backend, model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let fixtures = test_model::fixtures_dir();
        let image_path = fixtures.join("llamas.jpg");
        let bitmap = MtmdBitmap::from_file(&mtmd_ctx, image_path.to_str().unwrap()).unwrap();
        let input_text = MtmdInputText {
            text: "What is in this image? <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let n_positions = chunks.total_positions();
        let context_size = u32::try_from(n_positions + 256).unwrap_or(2048);
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(context_size));
        let llama_ctx = model.new_context(&backend, ctx_params).unwrap();
        let n_batch = i32::try_from(llama_ctx.n_batch()).unwrap();
        let result = chunks.eval_chunks(&mtmd_ctx, &llama_ctx, 0, 0, n_batch, false);

        assert!(result.is_ok());
    }

    fn eval_synthetic_bitmap(
        backend: &crate::llama_backend::LlamaBackend,
        model: &crate::model::LlamaModel,
        mtmd_ctx: &super::MtmdContext,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        let image_data = vec![128u8; (width as usize) * (height as usize) * 3];
        let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
        let input_text = MtmdInputText {
            text: "Describe: <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
        let n_positions = chunks.total_positions();
        let context_size = u32::try_from(n_positions + 256).unwrap_or(8192);
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(context_size));
        let llama_ctx = model.new_context(backend, ctx_params)?;
        let n_batch = i32::try_from(llama_ctx.n_batch())?;
        chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, false)?;

        Ok(())
    }

    #[test]
    #[serial]
    fn eval_chunks_with_varied_dimensions() {
        let (backend, model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();

        let test_dimensions: [(u32, u32); 4] = [(224, 224), (512, 512), (100, 500), (337, 421)];

        for (width, height) in test_dimensions {
            let result = eval_synthetic_bitmap(&backend, &model, &mtmd_ctx, width, height);

            assert!(
                result.is_ok(),
                "dimension {width}x{height} should succeed: {result:?}"
            );
        }
    }

    #[test]
    #[serial]
    fn decode_use_non_causal_returns_bool() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello world <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        let _non_causal = mtmd_ctx.decode_use_non_causal(&first_chunk);
    }

    #[test]
    #[serial]
    fn decode_use_mrope_returns_bool() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let _mrope = mtmd_ctx.decode_use_mrope();
    }

    #[test]
    #[serial]
    fn support_audio_returns_bool() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let _audio = mtmd_ctx.support_audio();
    }

    #[test]
    #[serial]
    fn get_audio_sample_rate_returns_option() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let _rate = mtmd_ctx.get_audio_sample_rate();
    }

    #[test]
    #[serial]
    fn encode_chunk_succeeds_for_image_chunk() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Describe: <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();

        for chunk_index in 0..chunks.len() {
            let chunk = chunks.get(chunk_index).unwrap();
            if chunk.chunk_type()
                == Ok(crate::mtmd::mtmd_input_chunk_type::MtmdInputChunkType::Image)
            {
                let result = mtmd_ctx.encode_chunk(&chunk);

                assert!(result.is_ok());

                return;
            }
        }
    }

    #[test]
    #[serial]
    fn tokenize_bitmap_count_mismatch_returns_error() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let input_text = MtmdInputText {
            text: "No media markers here".to_string(),
            add_special: true,
            parse_special: true,
        };
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let result = mtmd_ctx.tokenize(input_text, &[&bitmap]);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn eval_chunks_with_extreme_dimensions_does_not_crash() {
        let (backend, model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();

        let extreme_dimensions: [(u32, u32); 6] = [
            (1, 1),
            (7, 13),
            (3, 1000),
            (1000, 3),
            (1920, 1080),
            (4096, 4096),
        ];

        let mut any_reached_eval = false;

        for (width, height) in extreme_dimensions {
            match eval_synthetic_bitmap(&backend, &model, &mtmd_ctx, width, height) {
                Ok(()) => any_reached_eval = true,
                Err(error) => eprintln!("  {width}x{height} failed: {error}"),
            }
        }

        assert!(
            any_reached_eval,
            "at least one extreme dimension should reach eval_chunks"
        );
    }

    #[test]
    #[serial]
    fn init_from_file_with_null_byte_in_path_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let _ = &backend;
        let mtmd_params = crate::mtmd::MtmdContextParams::default();
        let result = crate::mtmd::MtmdContext::init_from_file("path\0null", &model, &mtmd_params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn tokenize_with_null_byte_in_text_returns_error() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let input_text = MtmdInputText {
            text: "text\0null".to_string(),
            add_special: true,
            parse_special: true,
        };
        let result = mtmd_ctx.tokenize(input_text, &[]);

        assert!(result.is_err());
    }
}

#[cfg(test)]
mod unit_tests {
    use super::check_encode_result;
    use super::tokenize_result_to_error;

    #[test]
    fn tokenize_result_bitmap_count_mismatch() {
        let error = tokenize_result_to_error(1);

        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn tokenize_result_image_preprocessing_error() {
        let error = tokenize_result_to_error(2);

        assert!(error.to_string().contains("Image preprocessing"));
    }

    #[test]
    fn tokenize_result_unknown_error() {
        let error = tokenize_result_to_error(42);

        assert!(error.to_string().contains("Unknown error: 42"));
    }

    #[test]
    fn check_encode_result_ok_for_zero() {
        assert!(check_encode_result(0).is_ok());
    }

    #[test]
    fn check_encode_result_error_for_nonzero() {
        let result = check_encode_result(5);

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Encode failed with code: 5")
        );
    }
}
