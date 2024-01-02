//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;

use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::timing::LlamaTimings;
use crate::token::data::LlamaTokenData;
use crate::token::LlamaToken;
use crate::{DecodeError};
use std::os::raw::c_int;
use std::ptr::NonNull;
use std::slice;

pub mod kv_cache;
pub mod params;
pub mod sample;

/// Safe wrapper around `llama_context`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<llama_cpp_sys_2::llama_context>,
    /// a reference to the contexts model.
    pub model: &'a LlamaModel,
    initialized_logits: Vec<i32>,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_2::llama_context>,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
        }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_decode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result as i32) {
            None => {
                self.initialized_logits = batch.initialized_logits.clone();
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_2::llama_reset_timings(self.context.as_ptr()) }
    }

    /// Rows: `n_tokens`
    /// Cols: `n_vocab`
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    pub fn logits_mut(&mut self, n_tokens: usize) -> &mut [f32] {
        let logits_ptr = unsafe { llama_cpp_sys_2::llama_get_logits(self.context.as_ptr()) };

        let n_vocab = usize::try_from(self.model.n_vocab()).expect("n_vocab should be positive");
        unsafe { slice::from_raw_parts_mut(logits_ptr, n_vocab * n_tokens) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_sys_2::llama_get_timings(self.context.as_ptr()) };
        LlamaTimings { timings }
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free(self.context.as_ptr()) }
    }
}
