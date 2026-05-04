use std::ptr::NonNull;

use crate::context::LlamaContext;

use super::mtmd_context::MtmdContext;
use super::mtmd_error::MtmdEvalError;
use super::mtmd_error::MtmdInputChunksError;
use super::mtmd_input_chunk::MtmdInputChunk;

const fn check_eval_result(result: i32) -> Result<(), MtmdEvalError> {
    if result == 0 {
        Ok(())
    } else {
        Err(MtmdEvalError::EvalFailure(result))
    }
}

/// Safe wrapper around `mtmd_input_chunks`.
///
/// This is a collection of input chunks created from tokenizing text and media.
/// The chunks represent the tokenized input that can be processed by the model,
/// with text chunks containing tokens and media chunks containing embeddings.
#[derive(Debug)]
pub struct MtmdInputChunks {
    /// Raw pointer to the underlying `mtmd_input_chunks`.
    pub chunks: NonNull<llama_cpp_bindings_sys::mtmd_input_chunks>,
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunksError::NullResult` if the underlying llama.cpp function
    /// returns null.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_bindings::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new().unwrap();
    /// assert_eq!(chunks.len(), 0);
    /// assert!(chunks.is_empty());
    /// ```
    pub fn new() -> Result<Self, MtmdInputChunksError> {
        let chunks = unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_init() };
        let chunks = NonNull::new(chunks).ok_or(MtmdInputChunksError::NullResult)?;

        Ok(Self { chunks })
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if chunks collection is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        NonNull::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            owned: false,
        })
    }

    /// Get total number of tokens across all chunks.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// Get total position count across all chunks.
    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Evaluate chunks using the multimodal context and LLAMA context.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEvalError::EvalFailure` if any encoding or decoding operation fails.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        n_past: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        let context_max_batch = llama_ctx.n_batch();

        if n_batch > 0 && n_batch.cast_unsigned() > context_max_batch {
            return Err(MtmdEvalError::BatchSizeExceedsContextLimit {
                requested: n_batch,
                context_max: context_max_batch,
            });
        }

        let mut new_n_past: llama_cpp_bindings_sys::llama_pos = 0;

        let result = unsafe {
            llama_cpp_bindings_sys::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
            )
        };

        check_eval_result(result)?;

        Ok(new_n_past)
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::MtmdInputChunks;

    #[test]
    fn new_creates_empty_chunks() {
        let chunks = MtmdInputChunks::new().unwrap();

        assert!(chunks.is_empty());
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let chunks = MtmdInputChunks::new().unwrap();

        assert!(chunks.get(0).is_none());
        assert!(chunks.get(999).is_none());
    }

    #[test]
    fn check_eval_result_ok_for_zero() {
        use super::check_eval_result;

        assert!(check_eval_result(0).is_ok());
    }

    #[test]
    fn check_eval_result_error_for_nonzero() {
        use super::check_eval_result;

        let result = check_eval_result(7);

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Eval failed with code: 7")
        );
    }
}
