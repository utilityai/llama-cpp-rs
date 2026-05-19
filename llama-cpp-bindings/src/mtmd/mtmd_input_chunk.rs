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

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data, which can be either text tokens,
/// image tokens, or audio tokens. The chunk type determines what kind of
/// data and operations are available.
#[derive(Debug)]
pub struct MtmdInputChunk {
    /// Raw pointer to the underlying `mtmd_input_chunk`.
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// Get the type of this chunk
    ///
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdInputChunkTypeError> {
        let chunk_type =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
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

    /// Get the number of tokens in this chunk
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk.
    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
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

    /// Create a copy of this chunk that you own.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::ChunkOperationFailed` if copying fails.
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::ChunkOperationFailed)?;

        Ok(Self { chunk, owned: true })
    }

    /// Evaluate this single chunk through the multimodal helper.
    ///
    /// Mirrors `MtmdInputChunks::eval_chunks` but for one chunk at a time, so
    /// callers can interleave per-chunk decode with per-chunk bookkeeping
    /// (token counting, marker state-machine replay) inside one loop instead
    /// of running the helper-level all-chunks eval and a separate ingest pass.
    ///
    /// Image chunks are decoded as one `llama_decode` call inside the helper,
    /// so their token count must fit in `n_batch`. When it would not, the
    /// binding refuses the call up front because the C-side
    /// `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` would otherwise abort
    /// the process.
    ///
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

        if matches!(self.chunk_type(), Ok(MtmdInputChunkType::Image))
            && i64::try_from(chunk_token_count).is_ok_and(|tokens| tokens > i64::from(n_batch))
        {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "image token counts and n_batch are model-bounded and fit in u32"
            )]
            return Err(MtmdEvalError::ImageChunkExceedsBatchSize(
                ImageChunkBatchSizeMismatch {
                    image_tokens: chunk_token_count as u32,
                    n_batch: n_batch as u32,
                },
            ));
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
            other => unreachable!(
                "llama_rs_mtmd_eval_chunk_single returned unrecognized status: {other}"
            ),
        }
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
    use super::tokens_from_raw_ptr;

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
}
