//! MTP (Multi-Token Prediction / NextN) speculative decoding.
//!
//! This binds llama.cpp's `common_speculative` API (type `draft-mtp`) through the
//! `llama_rs_speculative_*` C shim in `llama-cpp-sys-2`. MTP uses a single GGUF whose embedded
//! NextN head drafts tokens from the target model's own hidden state, so no separate draft model
//! is loaded.
//!
//! Usage:
//! 1. Create the target context with [`crate::model::LlamaModel::new_context`].
//! 2. Create the MTP draft context with [`crate::model::LlamaModel::new_mtp_context`].
//! 3. Create an [`MtpSpeculator`] over both.
//! 4. Prefill the target, then loop: [`MtpSpeculator::draft`] → decode the verify batch →
//!    [`MtpSpeculator::process`] → accept the matching prefix → [`MtpSpeculator::accept`].
//!
//! # Safety contract
//!
//! [`MtpSpeculator`] and the MTP draft context hold raw pointers to the target context inside
//! llama.cpp. The caller must keep the target context (and the model) alive at least as long as
//! the draft context and the speculator.

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
    /// A C++ exception (or internal decode failure) was caught at the shim boundary.
    #[error("c++ exception or decode failure at speculative shim boundary")]
    Exception,
    /// `common_speculative_init` returned null.
    #[error("failed to initialize MTP speculative context")]
    InitFailed,
    /// Creating the MTP draft context returned null.
    #[error("failed to create MTP draft context")]
    ContextCreationFailed,
}

pub(crate) fn status_to_result(status: llama_cpp_sys_2::llama_rs_status) -> Result<(), SpeculativeError> {
    match status {
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_OK => Ok(()),
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(SpeculativeError::InvalidArgument)
        }
        // ALLOCATION_FAILED, EXCEPTION, and any unknown code all surface as Exception.
        _ => Err(SpeculativeError::Exception),
    }
}

impl LlamaModel {
    /// Create the MTP draft context for `target`.
    ///
    /// The draft context runs the NextN graph on the **same model** and shares memory with the
    /// target context (`ctx_type = Mtp`, `ctx_other = target`). This mirrors the MTP setup in
    /// llama.cpp's server (`tools/server/server-context.cpp`).
    ///
    /// The returned context borrows the model. The caller must keep `target` alive at least as
    /// long as the returned context (it holds a raw pointer to `target`).
    pub fn new_mtp_context<'a>(
        &'a self,
        _: &LlamaBackend,
        target: &LlamaContext<'_>,
        mut params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, SpeculativeError> {
        params.context_params.ctx_type = LlamaContextType::Mtp.to_raw();
        params.context_params.ctx_other = target.context.as_ptr();

        let embeddings = params.embeddings();
        let ctx = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(self.model.as_ptr(), params.context_params)
        };
        let ctx = NonNull::new(ctx).ok_or(SpeculativeError::ContextCreationFailed)?;
        Ok(LlamaContext::new(self, ctx, embeddings))
    }
}

/// A safe handle to an MTP speculator (wraps a `common_speculative *`).
///
/// See the [module docs](self) for the lifetime/safety contract: the target and draft contexts
/// must outlive this speculator.
#[derive(Debug)]
pub struct MtpSpeculator {
    handle: NonNull<llama_cpp_sys_2::llama_rs_speculative>,
    n_max: i32,
}

impl MtpSpeculator {
    /// Initialize an MTP speculator over `target` (the real model context) and `draft` (the MTP
    /// context from [`LlamaModel::new_mtp_context`]).
    ///
    /// - `n_max` / `n_min`: max / min draft tokens per step.
    /// - `p_min`: minimum draft probability (0.0 = greedy drafting).
    /// - `backend_sampling`: offload draft sampling to the backend (recommended `true`).
    pub fn new(
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
        Ok(Self { handle, n_max })
    }

    /// True if the speculator needs target NextN embeddings extracted (always true for MTP).
    #[must_use]
    pub fn need_embd_nextn(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_need_embd_nextn(self.handle.as_ptr()) }
    }

    /// Optionally call once at the start of a generation with the already-decoded prompt tokens.
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

    /// Feed the just-decoded target verify batch so the speculator captures the hidden states it
    /// needs for the next draft. Call this after every `target.decode(...)`.
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_process(
                self.handle.as_ptr(),
                std::ptr::addr_of!(batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Generate draft tokens for `seq_id`, seeded from `id_last` at position `n_past`.
    ///
    /// `prompt` is the full list of tokens currently in the target context (some draft
    /// implementations consult it; MTP largely relies on the carried hidden state).
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

    /// Inform the speculator that `n_accepted` draft tokens were accepted by the target.
    /// `n_accepted` excludes the bonus/correction token.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_accept(self.handle.as_ptr(), seq_id, n_accepted)
        };
        status_to_result(status)
    }
}

impl Drop for MtpSpeculator {
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
