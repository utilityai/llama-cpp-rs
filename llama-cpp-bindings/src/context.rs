use std::ffi::c_void;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::context::params::LlamaContextParams;
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::timing::LlamaTimings;
use crate::token::LlamaToken;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaContextLoadError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError, LogitsError,
};

const fn check_lora_set_result(err_code: i32) -> Result<(), LlamaLoraAdapterSetError> {
    if err_code != 0 {
        return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
    }

    Ok(())
}

const fn check_lora_remove_result(err_code: i32) -> Result<(), LlamaLoraAdapterRemoveError> {
    if err_code != 0 {
        return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
    }

    Ok(())
}

fn new_context_with_model_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_new_context_with_model_status,
    out_ctx: *mut llama_cpp_bindings_sys::llama_context,
    out_error: *mut std::os::raw::c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_context>, LlamaContextLoadError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK => {
            NonNull::new(out_ctx).ok_or(LlamaContextLoadError::Unconstructible)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_VENDORED_RETURNED_NULL => {
            Err(LlamaContextLoadError::Unconstructible)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED => {
            Err(LlamaContextLoadError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(LlamaContextLoadError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_new_context_with_model returned unrecognized status {other}")
        }
    }
}

fn decode_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_decode_status,
    out_vendored_return_code: i32,
    out_error: *mut std::os::raw::c_char,
) -> Result<(), DecodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_VENDORED_RETURNED_NONZERO_CODE => {
            let code = NonZeroI32::new(out_vendored_return_code).unwrap_or_else(|| {
                unreachable!(
                    "llama_rs_decode reported a nonzero return code but the value was zero"
                )
            });
            Err(DecodeError::from(code))
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_OUT_OF_MEMORY => {
            Err(DecodeError::DecodeOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_COMPUTE_FAILED => Err(DecodeError::ComputeFailed),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(DecodeError::Reported { message })
        }
        other => unreachable!("llama_rs_decode returned unrecognized status {other}"),
    }
}

fn encode_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_encode_status,
    out_vendored_return_code: i32,
    out_error: *mut std::os::raw::c_char,
) -> Result<(), EncodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER => {
            Err(EncodeError::ModelHasNoEncoder)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_VENDORED_RETURNED_NONZERO_CODE => {
            let code = NonZeroI32::new(out_vendored_return_code).unwrap_or_else(|| {
                unreachable!(
                    "llama_rs_encode reported a nonzero return code but the value was zero"
                )
            });
            Err(EncodeError::from(code))
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OUT_OF_MEMORY => {
            Err(EncodeError::EncodeOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_COMPUTE_FAILED => Err(EncodeError::ComputeFailed),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(EncodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(EncodeError::Reported { message })
        }
        other => unreachable!("llama_rs_encode returned unrecognized status {other}"),
    }
}

fn token_index_within_context(token_index: i32, context_size: u32) -> Result<(), LogitsError> {
    if token_index >= 0 {
        let token_index_u32 =
            u32::try_from(token_index).map_err(LogitsError::TokenIndexOverflow)?;

        if context_size <= token_index_u32 {
            return Err(LogitsError::TokenIndexExceedsContext {
                token_index: token_index_u32,
                context_size,
            });
        }
    }

    Ok(())
}

unsafe fn logits_slice_from_raw_parts<'logits>(
    data: *const f32,
    n_vocab: i32,
) -> Result<&'logits [f32], LogitsError> {
    if data.is_null() {
        return Err(LogitsError::NullLogits);
    }

    let len = usize::try_from(n_vocab).map_err(LogitsError::VocabSizeOverflow)?;

    Ok(unsafe { slice::from_raw_parts(data, len) })
}

pub mod kv_cache;
pub mod kv_cache_type;
pub mod llama_attention_type;
pub mod llama_pooling_type;
pub mod llama_state_seq_flags;
pub mod load_seq_state_error;
pub mod load_session_error;
pub mod params;
pub mod rope_scaling_type;
pub mod save_seq_state_error;
pub mod save_session_error;
pub mod session;

unsafe extern "C" fn abort_callback_trampoline(data: *mut c_void) -> bool {
    let flag = unsafe { &*(data as *const AtomicBool) };

    flag.load(Ordering::Relaxed)
}

pub struct LlamaContext<'model> {
    pub context: NonNull<llama_cpp_bindings_sys::llama_context>,
    pub model: &'model LlamaModel,
    abort_flag: Option<Arc<AtomicBool>>,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    #[must_use]
    pub const fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_bindings_sys::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            abort_flag: None,
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
    }

    /// # Errors
    ///
    /// Returns [`LlamaContextLoadError`] when llama.cpp fails to allocate the context.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "LlamaContextParams may become non-trivially copyable upstream"
    )]
    pub fn from_model(
        model: &'model LlamaModel,
        _backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<Self, LlamaContextLoadError> {
        let context_params = params.context_params;
        let mut out_ctx: *mut llama_cpp_bindings_sys::llama_context = std::ptr::null_mut();
        let mut out_error: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_new_context_with_model(
                model.model.as_ptr(),
                context_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };
        let context = new_context_with_model_status_to_result(status, out_ctx, out_error)?;

        Ok(Self::new(model, context, params.embeddings()))
    }

    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_batch(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ubatch(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ctx(self.context.as_ptr()) }
    }

    #[expect(unsafe_code, reason = "required for FFI abort callback registration")]
    pub fn set_abort_flag(&mut self, flag: Arc<AtomicBool>) {
        let raw_ptr = Arc::as_ptr(&flag) as *mut c_void;
        self.abort_flag = Some(flag);

        unsafe {
            llama_cpp_bindings_sys::llama_set_abort_callback(
                self.context.as_ptr(),
                Some(abort_callback_trampoline),
                raw_ptr,
            );
        }
    }

    #[expect(unsafe_code, reason = "required for FFI abort callback deregistration")]
    pub fn clear_abort_callback(&mut self) {
        self.abort_flag = None;

        unsafe {
            llama_cpp_bindings_sys::llama_set_abort_callback(
                self.context.as_ptr(),
                None,
                std::ptr::null_mut(),
            );
        }
    }

    #[expect(unsafe_code, reason = "required for FFI synchronization call")]
    pub fn synchronize(&self) {
        unsafe { llama_cpp_bindings_sys::llama_synchronize(self.context.as_ptr()) }
    }

    #[expect(unsafe_code, reason = "required for FFI threadpool detachment")]
    pub fn detach_threadpool(&self) {
        unsafe { llama_cpp_bindings_sys::llama_detach_threadpool(self.context.as_ptr()) }
    }

    pub fn mark_logits_initialized(&mut self, token_index: i32) {
        self.initialized_logits = vec![token_index];
    }

    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let mut out_vendored_return_code: i32 = 0;
        let mut out_error: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_decode(
                self.context.as_ptr(),
                batch.llama_batch,
                &raw mut out_vendored_return_code,
                &raw mut out_error,
            )
        };
        decode_status_to_result(status, out_vendored_return_code, out_error)?;

        self.initialized_logits
            .clone_from(&batch.initialized_logits);

        Ok(())
    }

    /// # Errors
    ///
    /// - `EncodeError` if the encoding failed.
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let mut out_vendored_return_code: i32 = 0;
        let mut out_error: *mut std::os::raw::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_encode(
                self.context.as_ptr(),
                batch.llama_batch,
                &raw mut out_vendored_return_code,
                &raw mut out_error,
            )
        };
        encode_status_to_result(status, out_vendored_return_code, out_error)?;

        self.initialized_logits
            .clone_from(&batch.initialized_logits);

        Ok(())
    }

    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    pub fn embeddings_seq_ith(&self, sequence_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd = usize::try_from(self.model.n_embd())
            .map_err(EmbeddingsError::InvalidEmbeddingDimension)?;

        unsafe {
            let embedding = llama_cpp_bindings_sys::llama_get_embeddings_seq(
                self.context.as_ptr(),
                sequence_index,
            );

            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    pub fn embeddings_ith(&self, token_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd = usize::try_from(self.model.n_embd())
            .map_err(EmbeddingsError::InvalidEmbeddingDimension)?;

        unsafe {
            let embedding = llama_cpp_bindings_sys::llama_get_embeddings_ith(
                self.context.as_ptr(),
                token_index,
            );

            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn candidates(&self) -> Result<impl Iterator<Item = LlamaTokenData> + '_, LogitsError> {
        let logits = self.get_logits()?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn token_data_array(&self) -> Result<LlamaTokenDataArray, LogitsError> {
        Ok(LlamaTokenDataArray::from_iter(self.candidates()?, false))
    }

    /// # Errors
    /// Returns `LogitsError` if the logits pointer is null or `n_vocab` overflows.
    pub fn get_logits(&self) -> Result<&[f32], LogitsError> {
        let data = unsafe { llama_cpp_bindings_sys::llama_get_logits(self.context.as_ptr()) };

        unsafe { logits_slice_from_raw_parts(data, self.model.n_vocab()) }
    }

    /// # Errors
    /// Returns `LogitsError` if the token is not initialized or out of range.
    pub fn candidates_ith(
        &self,
        token_index: i32,
    ) -> Result<impl Iterator<Item = LlamaTokenData> + '_, LogitsError> {
        let logits = self.get_logits_ith(token_index)?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    /// # Errors
    /// Returns `LogitsError` if the token is not initialized or out of range.
    pub fn token_data_array_ith(
        &self,
        token_index: i32,
    ) -> Result<LlamaTokenDataArray, LogitsError> {
        Ok(LlamaTokenDataArray::from_iter(
            self.candidates_ith(token_index)?,
            false,
        ))
    }

    /// # Errors
    /// Returns `LogitsError` if the token is not initialized, out of range, or `n_vocab` overflows.
    pub fn get_logits_ith(&self, token_index: i32) -> Result<&[f32], LogitsError> {
        if !self.initialized_logits.contains(&token_index) {
            return Err(LogitsError::TokenNotInitialized(token_index));
        }

        token_index_within_context(token_index, self.n_ctx())?;

        let data = unsafe {
            llama_cpp_bindings_sys::llama_get_logits_ith(self.context.as_ptr(), token_index)
        };
        let len = usize::try_from(self.model.n_vocab()).map_err(LogitsError::VocabSizeOverflow)?;

        Ok(unsafe { slice::from_raw_parts(data, len) })
    }

    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_perf_context_reset(self.context.as_ptr()) }
    }

    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_bindings_sys::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let mut adapters = [adapter.lora_adapter.as_ptr()];
        let mut scales = [scale];
        let err_code = unsafe {
            llama_cpp_bindings_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                adapters.as_mut_ptr(),
                1,
                scales.as_mut_ptr(),
            )
        };
        check_lora_set_result(err_code)?;

        log::debug!("Set lora adapter");
        Ok(())
    }

    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_bindings_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        check_lora_remove_result(err_code)?;

        log::debug!("Remove lora adapter");
        Ok(())
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::DecodeError;
    use crate::EncodeError;
    use crate::LlamaContextLoadError;
    use crate::LlamaLoraAdapterRemoveError;
    use crate::LlamaLoraAdapterSetError;
    use crate::LogitsError;

    use super::{
        check_lora_remove_result, check_lora_set_result, decode_status_to_result,
        encode_status_to_result, logits_slice_from_raw_parts,
        new_context_with_model_status_to_result, token_index_within_context,
    };

    #[test]
    fn check_lora_set_result_ok_for_zero() {
        assert!(check_lora_set_result(0).is_ok());
    }

    #[test]
    fn check_lora_set_result_error_for_nonzero() {
        let result = check_lora_set_result(-1);

        assert_eq!(result, Err(LlamaLoraAdapterSetError::ErrorResult(-1)));
    }

    #[test]
    fn check_lora_remove_result_ok_for_zero() {
        assert!(check_lora_remove_result(0).is_ok());
    }

    #[test]
    fn check_lora_remove_result_error_for_nonzero() {
        let result = check_lora_remove_result(-1);

        assert_eq!(result, Err(LlamaLoraAdapterRemoveError::ErrorResult(-1)));
    }

    #[test]
    fn new_context_ok_with_null_ctx_maps_unconstructible() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(LlamaContextLoadError::Unconstructible));
    }

    #[test]
    fn new_context_vendored_returned_null_maps_unconstructible() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_VENDORED_RETURNED_NULL,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(LlamaContextLoadError::Unconstructible));
    }

    #[test]
    fn new_context_allocation_failed_maps_not_enough_memory() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(LlamaContextLoadError::NotEnoughMemory));
    }

    #[test]
    fn new_context_cxx_exception_maps_reported() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_VENDORED_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(LlamaContextLoadError::Reported {
                message: "unknown error".to_owned(),
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_new_context_with_model returned unrecognized status")]
    fn new_context_unrecognized_status_panics() {
        let _result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::llama_rs_new_context_with_model_status::MAX,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }

    #[test]
    fn decode_nonzero_code_maps_from_code() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_VENDORED_RETURNED_NONZERO_CODE,
            1,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::NoKvCacheSlot));
    }

    #[test]
    fn decode_out_of_memory_maps_decode_out_of_memory() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_OUT_OF_MEMORY,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::DecodeOutOfMemory));
    }

    #[test]
    fn decode_compute_failed_maps_compute_failed() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_COMPUTE_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::ComputeFailed));
    }

    #[test]
    fn decode_allocation_failed_maps_not_enough_memory() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::NotEnoughMemory));
    }

    #[test]
    fn decode_cxx_exception_maps_reported() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_VENDORED_THREW_CXX_EXCEPTION,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(DecodeError::Reported {
                message: "unknown error".to_owned(),
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_decode reported a nonzero return code")]
    fn decode_nonzero_code_with_zero_value_panics() {
        let _result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_VENDORED_RETURNED_NONZERO_CODE,
            0,
            std::ptr::null_mut(),
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_decode returned unrecognized status")]
    fn decode_unrecognized_status_panics() {
        let _result = decode_status_to_result(
            llama_cpp_bindings_sys::llama_rs_decode_status::MAX,
            0,
            std::ptr::null_mut(),
        );
    }

    #[test]
    fn encode_model_has_no_encoder_maps_model_has_no_encoder() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::ModelHasNoEncoder));
    }

    #[test]
    fn encode_nonzero_code_maps_from_code() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_VENDORED_RETURNED_NONZERO_CODE,
            1,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::NoKvCacheSlot));
    }

    #[test]
    fn encode_out_of_memory_maps_encode_out_of_memory() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OUT_OF_MEMORY,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::EncodeOutOfMemory));
    }

    #[test]
    fn encode_compute_failed_maps_compute_failed() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_COMPUTE_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::ComputeFailed));
    }

    #[test]
    fn encode_allocation_failed_maps_not_enough_memory() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::NotEnoughMemory));
    }

    #[test]
    fn encode_cxx_exception_maps_reported() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_VENDORED_THREW_CXX_EXCEPTION,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(EncodeError::Reported {
                message: "unknown error".to_owned(),
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_encode reported a nonzero return code")]
    fn encode_nonzero_code_with_zero_value_panics() {
        let _result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_VENDORED_RETURNED_NONZERO_CODE,
            0,
            std::ptr::null_mut(),
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_encode returned unrecognized status")]
    fn encode_unrecognized_status_panics() {
        let _result = encode_status_to_result(
            llama_cpp_bindings_sys::llama_rs_encode_status::MAX,
            0,
            std::ptr::null_mut(),
        );
    }

    #[test]
    fn token_index_beyond_context_size_maps_exceeds_context() {
        let result = token_index_within_context(5, 4);

        assert_eq!(
            result,
            Err(LogitsError::TokenIndexExceedsContext {
                token_index: 5,
                context_size: 4,
            })
        );
    }

    #[test]
    fn token_index_within_context_size_is_ok() {
        assert!(token_index_within_context(2, 4).is_ok());
    }

    #[test]
    fn token_index_negative_skips_context_check() {
        assert!(token_index_within_context(-1, 4).is_ok());
    }

    #[test]
    fn logits_slice_from_null_data_maps_null_logits() {
        let result = unsafe { logits_slice_from_raw_parts(std::ptr::null(), 4) };

        assert_eq!(result, Err(LogitsError::NullLogits));
    }

    #[test]
    fn logits_slice_from_negative_vocab_maps_vocab_size_overflow() {
        let logit_value = 0.0_f32;
        let result = unsafe { logits_slice_from_raw_parts(&raw const logit_value, -1) };

        let conversion_error = usize::try_from(-1_i32).unwrap_err();

        assert_eq!(
            result,
            Err(LogitsError::VocabSizeOverflow(conversion_error))
        );
    }
}
