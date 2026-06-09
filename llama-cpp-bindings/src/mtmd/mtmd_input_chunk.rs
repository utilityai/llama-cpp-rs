use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr::NonNull;
use std::slice;

use crate::context::LlamaContext;
use crate::ffi_error_reader::read_and_free_cpp_error;
use crate::token::LlamaToken;

use super::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;
use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk_error::MtmdInputChunkError;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

/// # Safety
///
/// `tokens_ptr` must point to at least `n_tokens` valid `llama_token` values
/// that remain valid for the lifetime `'chunk`.
const unsafe fn tokens_from_raw_ptr<'chunk>(
    tokens_ptr: *const llama_cpp_bindings_sys::llama_token,
    n_tokens: usize,
) -> Option<&'chunk [LlamaToken]> {
    if tokens_ptr.is_null() || n_tokens == 0 {
        None
    } else {
        unsafe {
            Some(slice::from_raw_parts(
                tokens_ptr.cast::<LlamaToken>(),
                n_tokens,
            ))
        }
    }
}

fn eval_chunk_single_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single_status,
    final_position: llama_cpp_bindings_sys::llama_pos,
    out_vendored_return_code: i32,
    out_error: *mut c_char,
) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK => Ok(final_position),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_RETURNED_NONZERO_CODE => {
            Err(MtmdEvalError::EvalFailed {
                code: out_vendored_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdEvalError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(out_error) };
            Err(MtmdEvalError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_mtmd_eval_chunk_single returned unrecognized status: {other}")
        }
    }
}

fn image_chunk_batch_size_error(
    is_image_chunk: bool,
    chunk_token_count: usize,
    n_batch: i32,
) -> Option<MtmdEvalError> {
    if is_image_chunk
        && i64::try_from(chunk_token_count).is_ok_and(|tokens| tokens > i64::from(n_batch))
    {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "image token counts and n_batch are model-bounded and fit in u32"
        )]
        return Some(MtmdEvalError::ImageChunkExceedsBatchSize(
            ImageChunkBatchSizeMismatch {
                image_tokens: chunk_token_count as u32,
                n_batch: n_batch as u32,
            },
        ));
    }

    None
}

#[derive(Debug)]
pub struct MtmdInputChunk {
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdInputChunkTypeError> {
        let chunk_type =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::try_from(chunk_type)
    }

    #[must_use]
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != Ok(MtmdInputChunkType::Text) {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        unsafe { tokens_from_raw_ptr(tokens_ptr, n_tokens) }
    }

    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::ChunkOperationFailed` if copying fails.
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::ChunkOperationFailed)?;

        Ok(Self { chunk, owned: true })
    }

    /// # Errors
    ///
    /// Returns [`MtmdEvalError::ImageChunkExceedsBatchSize`] when this is an
    /// image chunk whose token count exceeds `n_batch`. Returns
    /// [`MtmdEvalError::EvalFailure`] if the underlying encode or decode step
    /// fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        let chunk_token_count = self.n_tokens();

        if let Some(error) = image_chunk_batch_size_error(
            matches!(self.chunk_type(), Ok(MtmdInputChunkType::Image)),
            chunk_token_count,
            n_batch,
        ) {
            return Err(error);
        }

        let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;
        let mut out_vendored_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunk.as_ptr(),
                start_position,
                seq_id,
                n_batch,
                logits_last,
                &raw mut final_position,
                &raw mut out_vendored_return_code,
                &raw mut out_error,
            )
        };

        eval_chunk_single_status_to_result(
            status,
            final_position,
            out_vendored_return_code,
            out_error,
        )
    }
}

impl Drop for MtmdInputChunk {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::eval_chunk_single_status_to_result;
    use super::image_chunk_batch_size_error;
    use super::tokens_from_raw_ptr;
    use crate::mtmd::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert!(unsafe { tokens_from_raw_ptr(std::ptr::null(), 5) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::llama_token = 42;
        assert!(unsafe { tokens_from_raw_ptr(&raw const token, 0) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_some_for_valid() {
        let tokens: [llama_cpp_bindings_sys::llama_token; 2] = [1, 2];
        let result = unsafe { tokens_from_raw_ptr(tokens.as_ptr(), 2) };

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn eval_chunk_single_status_ok_returns_final_position() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK,
            7,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Ok(7));
    }

    #[test]
    fn eval_chunk_single_status_nonzero_code_maps_to_eval_failed() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_RETURNED_NONZERO_CODE,
            0,
            -3,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEvalError::EvalFailed { code: -3 }));
    }

    #[test]
    fn eval_chunk_single_status_allocation_failed_maps_to_not_enough_memory() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEvalError::NotEnoughMemory));
    }

    #[test]
    fn eval_chunk_single_status_cxx_exception_reports_unknown_error_for_null() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_THREW_CXX_EXCEPTION,
            0,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdEvalError::Reported {
                message: "unknown error".to_string()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_mtmd_eval_chunk_single returned unrecognized status")]
    fn eval_chunk_single_status_unrecognized_panics() {
        let _ = eval_chunk_single_status_to_result(u32::MAX, 0, 0, std::ptr::null_mut());
    }

    #[test]
    fn image_chunk_over_batch_size_reports_mismatch() {
        let error = image_chunk_batch_size_error(true, 9, 4);

        assert_eq!(
            error,
            Some(MtmdEvalError::ImageChunkExceedsBatchSize(
                ImageChunkBatchSizeMismatch {
                    image_tokens: 9,
                    n_batch: 4,
                }
            ))
        );
    }

    #[test]
    fn non_image_chunk_never_reports_mismatch() {
        assert!(image_chunk_batch_size_error(false, 9, 4).is_none());
    }

    #[test]
    fn image_chunk_within_batch_size_reports_no_mismatch() {
        assert!(image_chunk_batch_size_error(true, 4, 4).is_none());
    }
}
