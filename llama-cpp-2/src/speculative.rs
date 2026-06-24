//! MTP (Multi-Token Prediction / `NextN`) speculative decoding via llama.cpp's `common_speculative`
//! (`draft-mtp`). A single GGUF's embedded `NextN` head drafts tokens from the target model's own
//! hidden state, so no separate draft model is loaded.
//!
//! The target and draft contexts must outlive the [`MtpSpeculator`], which holds raw pointers to
//! them inside llama.cpp.

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::context::params::{LlamaContextParams, LlamaContextType};
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// Errors from MTP speculative decoding.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeError {
    /// A shim call rejected its arguments (e.g. null handle).
    #[error("invalid argument passed to speculative shim")]
    InvalidArgument,
    /// The draft output buffer was too small for the produced draft.
    #[error("draft buffer too small (needed {needed}, had {had})")]
    BufferTooSmall {
        /// number of tokens the draft produced
        needed: usize,
        /// capacity of the buffer that was provided
        had: usize,
    },
    /// A C++ exception or internal decode failure was caught at the shim boundary.
    #[error("c++ exception or decode failure at speculative shim boundary")]
    Exception,
    /// `common_speculative_init` returned null.
    #[error("failed to initialize MTP speculative context")]
    InitFailed,
    /// Creating the MTP draft context returned null.
    #[error("failed to create MTP draft context")]
    ContextCreationFailed,
}

pub(crate) fn status_to_result(
    status: llama_cpp_sys_2::llama_rs_status,
) -> Result<(), SpeculativeError> {
    match status {
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_OK => Ok(()),
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(SpeculativeError::InvalidArgument)
        }
        _ => Err(SpeculativeError::Exception),
    }
}

impl LlamaModel {
    /// Create the MTP draft context for `target`. It runs the `NextN` graph on the same model with
    /// `ctx_type = Mtp` and `ctx_other = target`. The returned context borrows the model.
    ///
    /// # Safety
    ///
    /// The draft context stores a raw pointer to `target` (`ctx_other`) inside llama.cpp, but the
    /// returned context's lifetime is tied only to the model, not to `target`. The caller must keep
    /// `target` alive and valid for at least as long as the returned context, otherwise any use of
    /// the draft context (e.g. via [`MtpSpeculator`]) dereferences freed memory. A shared borrow of
    /// `target` cannot express this, since `target` must still be mutated (e.g. via `decode`) while
    /// the draft context lives, so the contract is the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeError::ContextCreationFailed`] if llama.cpp fails to create the context.
    pub unsafe fn new_mtp_context<'a>(
        &'a self,
        _: &LlamaBackend,
        target: &LlamaContext<'_>,
        mut params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, SpeculativeError> {
        params.context_params.ctx_type = LlamaContextType::Mtp.to_raw();
        params.context_params.ctx_other = target.context.as_ptr();

        let embeddings = params.embeddings();
        let ctx = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(
                self.model.as_ptr(),
                params.context_params,
            )
        };
        let ctx = NonNull::new(ctx).ok_or(SpeculativeError::ContextCreationFailed)?;
        Ok(LlamaContext::new(self, ctx, embeddings))
    }
}

/// A handle to an MTP speculator (wraps a `common_speculative *`). The lifetime `'a` is tied to the
/// target and draft contexts, which it holds raw pointers to inside llama.cpp and which must
/// therefore outlive it.
#[derive(Debug)]
pub struct MtpSpeculator<'a> {
    handle: NonNull<llama_cpp_sys_2::llama_rs_speculative>,
    n_max: i32,
    _contexts: PhantomData<&'a ()>,
}

impl MtpSpeculator<'_> {
    /// Initialize an MTP speculator over `target` and `draft` (from [`LlamaModel::new_mtp_context`]).
    /// `n_max`/`n_min` bound the draft length, `p_min` is the minimum draft probability (0.0 =
    /// greedy), and `backend_sampling` offloads draft sampling to the backend.
    ///
    /// # Safety
    ///
    /// The speculator stores raw pointers to `target` and `draft` inside llama.cpp; both contexts
    /// must remain alive and valid for the speculator's lifetime `'a`. A shared borrow cannot
    /// express this (the contexts are still mutated, e.g. via `decode`), so the contract is the
    /// caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeError::InitFailed`] if `common_speculative_init` returns null.
    pub unsafe fn new(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        n_max: i32,
        n_min: i32,
        p_min: f32,
        backend_sampling: bool,
    ) -> Result<Self, SpeculativeError> {
        let handle = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_init_mtp(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                n_max,
                n_min,
                p_min,
                backend_sampling,
            )
        };
        let handle = NonNull::new(handle).ok_or(SpeculativeError::InitFailed)?;
        Ok(Self {
            handle,
            n_max,
            _contexts: PhantomData,
        })
    }

    /// Whether the speculator needs target `NextN` embeddings extracted (always true for MTP).
    #[must_use]
    pub fn need_embd_nextn(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_need_embd_nextn(self.handle.as_ptr()) }
    }

    /// Optionally call once at the start of a generation with the already-decoded prompt tokens.
    ///
    /// # Errors
    ///
    /// Returns a [`SpeculativeError`] if the shim call fails.
    pub fn begin(&mut self, seq_id: i32, prompt: &[LlamaToken]) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_begin(
                self.handle.as_ptr(),
                seq_id,
                prompt.as_ptr().cast(),
                prompt.len(),
            )
        };
        status_to_result(status)
    }

    /// Feed the just-decoded target verify batch so the speculator captures the hidden states for
    /// the next draft. Call after every `target.decode(...)`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpeculativeError`] if the shim call or the internal draft-context decode fails.
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_process(
                self.handle.as_ptr(),
                std::ptr::addr_of!(batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Generate draft tokens for `seq_id`, seeded from `id_last` at position `n_past`. `prompt` is
    /// the tokens currently in the target context.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeError::BufferTooSmall`] if the draft exceeds the configured `n_max`, or
    /// another [`SpeculativeError`] if the shim call fails.
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
        prompt: &[LlamaToken],
    ) -> Result<Vec<LlamaToken>, SpeculativeError> {
        let cap = self.n_max.max(1) as usize;
        let mut buf = vec![LlamaToken(0); cap];
        let mut out_len: usize = 0;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_draft(
                self.handle.as_ptr(),
                seq_id,
                n_past,
                id_last.0,
                prompt.as_ptr().cast(),
                prompt.len(),
                buf.as_mut_ptr().cast(),
                buf.len(),
                std::ptr::addr_of_mut!(out_len),
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_BUFFER_TOO_SMALL {
            return Err(SpeculativeError::BufferTooSmall {
                needed: out_len,
                had: cap,
            });
        }
        status_to_result(status)?;
        buf.truncate(out_len);
        Ok(buf)
    }

    /// Inform the speculator that `n_accepted` draft tokens were accepted (excludes the bonus token).
    ///
    /// # Errors
    ///
    /// Returns a [`SpeculativeError`] if the shim call fails.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_accept(self.handle.as_ptr(), seq_id, n_accepted)
        };
        status_to_result(status)
    }
}

impl Drop for MtpSpeculator<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_free(self.handle.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::{status_to_result, SpeculativeError};

    #[test]
    fn status_mapping() {
        assert!(status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_OK).is_ok());
        assert!(matches!(
            status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT),
            Err(SpeculativeError::InvalidArgument)
        ));
        assert!(matches!(
            status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION),
            Err(SpeculativeError::Exception)
        ));
        assert!(matches!(
            status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED),
            Err(SpeculativeError::Exception)
        ));
    }
}
