//! Safe wrapper around `llama_context`.

use std::ffi::c_void;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::timing::LlamaTimings;
use crate::token::LlamaToken;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
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

pub mod kv_cache;
pub mod llama_state_seq_flags;
pub mod load_seq_state_error;
pub mod load_session_error;
pub mod params;
pub mod save_seq_state_error;
pub mod save_session_error;
pub mod session;

unsafe extern "C" fn abort_callback_trampoline(data: *mut c_void) -> bool {
    let flag = unsafe { &*(data as *const AtomicBool) };

    flag.load(Ordering::Relaxed)
}

/// Safe wrapper around `llama_context`.
pub struct LlamaContext<'model> {
    /// Raw pointer to the underlying `llama_context`.
    pub context: NonNull<llama_cpp_bindings_sys::llama_context>,
    /// A reference to the context's model.
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
    /// Wraps existing raw pointers into a new `LlamaContext`.
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

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to [`Self::n_ubatch`].
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to [`Self::n_batch`].
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Sets an abort flag that llama.cpp checks during computation.
    ///
    /// When the flag is set to `true`, any in-progress `decode()` call will
    /// abort and return `DecodeError::Aborted`. The `Arc` is stored internally
    /// to ensure the flag outlives the callback registration.
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

    /// Clears the abort callback so that decode calls are no longer interruptible.
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

    /// Waits for all pending backend operations to complete.
    ///
    /// Must be called before freeing the context to prevent hangs
    /// during resource cleanup.
    #[expect(unsafe_code, reason = "required for FFI synchronization call")]
    pub fn synchronize(&self) {
        unsafe { llama_cpp_bindings_sys::llama_synchronize(self.context.as_ptr()) }
    }

    /// Detaches the threadpool from the context.
    ///
    /// Must be called before freeing the context to prevent threadpool
    /// workers from accessing freed resources.
    #[expect(unsafe_code, reason = "required for FFI threadpool detachment")]
    pub fn detach_threadpool(&self) {
        unsafe { llama_cpp_bindings_sys::llama_detach_threadpool(self.context.as_ptr()) }
    }

    /// Marks a logit index as initialized so it can be read via
    /// `get_logits_ith`. Use after external decode operations (like
    /// `eval_chunks`) that bypass the Rust `decode()` method.
    pub fn mark_logits_initialized(&mut self, token_index: i32) {
        self.initialized_logits = vec![token_index];
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result = unsafe {
            llama_cpp_bindings_sys::llama_decode(self.context.as_ptr(), batch.llama_batch)
        };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_encode(self.context.as_ptr(), batch.llama_batch)
        };

        self.handle_encode_result(status, batch)
    }

    fn handle_encode_result(
        &mut self,
        status: llama_cpp_bindings_sys::llama_rs_status,
        batch: &mut LlamaBatch,
    ) -> Result<(), EncodeError> {
        if crate::status_is_ok(status) {
            self.initialized_logits
                .clone_from(&batch.initialized_logits);

            Ok(())
        } else {
            Err(EncodeError::from(
                NonZeroI32::new(crate::status_to_i32(status))
                    .unwrap_or(NonZeroI32::new(1).expect("1 is non-zero")),
            ))
        }
    }

    /// Get the embeddings for the given sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
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

    /// Get the embeddings for the given token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
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

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn candidates(&self) -> Result<impl Iterator<Item = LlamaTokenData> + '_, LogitsError> {
        let logits = self.get_logits()?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    /// Get the token data array for the last token in the context.
    ///
    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn token_data_array(&self) -> Result<LlamaTokenDataArray, LogitsError> {
        Ok(LlamaTokenDataArray::from_iter(self.candidates()?, false))
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Errors
    /// Returns `LogitsError` if the logits pointer is null or `n_vocab` overflows.
    pub fn get_logits(&self) -> Result<&[f32], LogitsError> {
        let data = unsafe { llama_cpp_bindings_sys::llama_get_logits(self.context.as_ptr()) };

        if data.is_null() {
            return Err(LogitsError::NullLogits);
        }

        let len = usize::try_from(self.model.n_vocab()).map_err(LogitsError::VocabSizeOverflow)?;

        Ok(unsafe { slice::from_raw_parts(data, len) })
    }

    /// Get the logits for the ith token in the context.
    ///
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

    /// Get the token data array for the ith token in the context.
    ///
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

    /// Get the logits for the ith token in the context.
    ///
    /// # Errors
    /// Returns `LogitsError` if the token is not initialized, out of range, or `n_vocab` overflows.
    pub fn get_logits_ith(&self, token_index: i32) -> Result<&[f32], LogitsError> {
        if !self.initialized_logits.contains(&token_index) {
            return Err(LogitsError::TokenNotInitialized(token_index));
        }

        if token_index >= 0 {
            let token_index_u32 =
                u32::try_from(token_index).map_err(LogitsError::TokenIndexOverflow)?;

            if self.n_ctx() <= token_index_u32 {
                return Err(LogitsError::TokenIndexExceedsContext {
                    token_index: token_index_u32,
                    context_size: self.n_ctx(),
                });
            }
        }

        let data = unsafe {
            llama_cpp_bindings_sys::llama_get_logits_ith(self.context.as_ptr(), token_index)
        };
        let len = usize::try_from(self.model.n_vocab()).map_err(LogitsError::VocabSizeOverflow)?;

        Ok(unsafe { slice::from_raw_parts(data, len) })
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_bindings_sys::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
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

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove all lora adapters.
    ///
    /// Note: The upstream API now replaces all adapters at once via
    /// `llama_set_adapters_lora`. This clears all adapters from the context.
    ///
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

        tracing::debug!("Remove lora adapter");
        Ok(())
    }

    /// Print a breakdown of per-device memory use to the default logger.
    pub fn print_memory_breakdown(&self) {
        unsafe { llama_cpp_bindings_sys::llama_memory_breakdown_print(self.context.as_ptr()) }
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::LlamaLoraAdapterRemoveError;
    use crate::LlamaLoraAdapterSetError;

    use super::{check_lora_remove_result, check_lora_set_result};

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
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::llama_batch::LlamaBatch;
    use crate::model::AddBos;
    use crate::test_model;

    #[test]
    #[serial]
    fn context_creation_and_properties() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        assert!(context.n_ctx() > 0);
        assert!(context.n_batch() > 0);
        assert!(context.n_ubatch() > 0);
    }

    #[test]
    #[serial]
    fn decode_and_get_logits() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let decode_result = context.decode(&mut batch);
        assert!(decode_result.is_ok());

        let logits = context.get_logits().unwrap();
        assert!(!logits.is_empty());
    }

    #[test]
    #[serial]
    fn timings_work() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        context.reset_timings();
        let timings = context.timings();
        assert!(timings.t_start_ms() >= 0.0);
    }

    #[test]
    #[serial]
    fn token_data_array_has_entries_after_decode() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let token_data_array = context.token_data_array().unwrap();

        assert!(!token_data_array.data.is_empty());
    }

    #[test]
    #[serial]
    fn get_logits_ith_returns_valid_slice() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let logits = context.get_logits_ith(last_index).unwrap();

        assert_eq!(logits.len(), model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn token_data_array_ith_returns_valid_data() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let token_data_array = context.token_data_array_ith(last_index).unwrap();

        assert_eq!(token_data_array.data.len(), model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_error_when_embeddings_disabled() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(false);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_ith(0);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_error_when_embeddings_disabled() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(false);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_seq_ith(0);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn candidates_returns_n_vocab_entries() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let count = context.candidates().unwrap().count();

        assert_eq!(count, model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn debug_format_contains_struct_name() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let debug_output = format!("{context:?}");

        assert!(debug_output.contains("LlamaContext"));
    }

    #[test]
    #[serial]
    fn decode_with_embeddings_enabled() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_valid_embeddings() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let embeddings = context.embeddings_seq_ith(0).unwrap();

        assert_eq!(embeddings.len(), model.n_embd() as usize);
    }

    #[test]
    #[serial]
    fn multi_sequence_embeddings_returns_one_embedding_per_sequence() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_n_seq_max(4)
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let inputs = [
            "alpha is here",
            "beta runs fast",
            "gamma waits",
            "delta jumps",
        ];
        let mut batch = LlamaBatch::new(64, 4).unwrap();

        for (sequence_index, text) in inputs.iter().enumerate() {
            let tokens = model.str_to_token(text, AddBos::Always).unwrap();
            let sequence_id = i32::try_from(sequence_index).unwrap();

            batch.add_sequence(&tokens, sequence_id, true).unwrap();
        }

        context.decode(&mut batch).unwrap();

        let n_embd = model.n_embd() as usize;
        let mut collected: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());

        for sequence_index in 0..inputs.len() {
            let sequence_id = i32::try_from(sequence_index).unwrap();
            let embedding = context.embeddings_seq_ith(sequence_id).unwrap();

            assert_eq!(
                embedding.len(),
                n_embd,
                "sequence {sequence_index} embedding length mismatch"
            );

            collected.push(embedding.to_vec());
        }

        for (left_index, left) in collected.iter().enumerate() {
            for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
                assert_ne!(
                    left, right,
                    "embedding for sequence {left_index} must differ from sequence {right_index}",
                );
            }
        }
    }

    /// Reproduces paddler's embedding batching loop exactly with the document strings, batch
    /// shape, and iteration pattern from the failing harness test
    /// `agent_embedding_batch_distribution_independent_of_context_size`. A `LlamaBatch` is
    /// allocated once with `n_tokens=64` and `n_seq_max=4`, then reused across two iterations
    /// of two sequences each (because the four ~22-token docs do not all fit in one
    /// 64-token window). Per iteration: `add_sequence` for each doc, `clear_kv_cache`,
    /// `decode`, `embeddings_seq_ith` for each filled slot, `batch.clear()`. Every iteration
    /// must yield distinct, non-empty embeddings — including iterations after the first.
    #[test]
    #[serial]
    fn embeddings_returns_distinct_values_when_reused_batch_has_extra_capacity() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_n_seq_max(4)
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let iterations = [
            [
                "This is the first document with enough content to contribute meaningfully to the batch size calculation",
                "This is the second document that should be processed in a potentially different batch from the first",
            ],
            [
                "This is the third document adding more content to ensure the total exceeds the configured chunk limit",
                "This is the fourth document which should demonstrate that batching distributes across agent requests",
            ],
        ];

        let n_embd = model.n_embd() as usize;
        let mut batch = LlamaBatch::new(64, 4).unwrap();
        let mut collected: Vec<Vec<f32>> = Vec::new();

        for iteration_inputs in iterations {
            for (sequence_index, text) in iteration_inputs.iter().enumerate() {
                let tokens = model.str_to_token(text, AddBos::Always).unwrap();
                let sequence_id = i32::try_from(sequence_index).unwrap();

                batch.add_sequence(&tokens, sequence_id, true).unwrap();
            }

            context.clear_kv_cache();
            context.decode(&mut batch).unwrap();

            for sequence_index in 0..iteration_inputs.len() {
                let sequence_id = i32::try_from(sequence_index).unwrap();
                let embedding = context.embeddings_seq_ith(sequence_id).unwrap();

                assert_eq!(
                    embedding.len(),
                    n_embd,
                    "iteration sequence {sequence_index} embedding length mismatch"
                );

                collected.push(embedding.to_vec());
            }

            batch.clear();
        }

        assert_eq!(
            collected.len(),
            iterations.iter().flatten().count(),
            "expected one embedding per input across every iteration"
        );

        for (left_index, left) in collected.iter().enumerate() {
            for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
                assert_ne!(
                    left, right,
                    "embedding {left_index} must differ from embedding {right_index} across reused-batch iterations",
                );
            }
        }
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_valid_embeddings() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let embeddings = context.embeddings_ith(last_index).unwrap();

        assert_eq!(embeddings.len(), model.n_embd() as usize);
    }

    #[test]
    #[serial]
    fn candidates_ith_returns_n_vocab_entries() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let count = context.candidates_ith(last_index).unwrap().count();

        assert_eq!(count, model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn lora_adapter_remove_succeeds_with_no_adapters() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let mut adapter = crate::model::LlamaLoraAdapter {
            lora_adapter: std::ptr::NonNull::dangling(),
        };

        let result = context.lora_adapter_remove(&mut adapter);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn encode_on_non_encoder_model_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.encode(&mut batch);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn lora_adapter_set_with_dangling_pointer_succeeds_or_errors() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let mut adapter = crate::model::LlamaLoraAdapter {
            lora_adapter: std::ptr::NonNull::dangling(),
        };

        let result = context.lora_adapter_set(&mut adapter, 1.0);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_null_embedding_error_for_non_embedding_token() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_ith(999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_null_embedding_error_for_invalid_seq() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.embeddings_seq_ith(999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn decode_empty_batch_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn encode_succeeds_with_encoder_model() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_encoder_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Never).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.encode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn handle_encode_result_ok_updates_logits() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, true).unwrap();

        let result =
            context.handle_encode_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, &mut batch);

        assert!(result.is_ok());
        assert!(!context.initialized_logits.is_empty());
    }

    #[test]
    #[serial]
    fn set_abort_flag_aborts_decode() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::Ordering;

        use crate::DecodeError;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let abort_flag = Arc::new(AtomicBool::new(true));
        context.set_abort_flag(abort_flag.clone());

        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.decode(&mut batch);

        assert_eq!(result, Err(DecodeError::Aborted));
    }

    #[test]
    #[serial]
    fn set_abort_flag_false_allows_decode() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let abort_flag = Arc::new(AtomicBool::new(false));
        context.set_abort_flag(abort_flag);

        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn clear_abort_callback_allows_decode_with_flag_true() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let abort_flag = Arc::new(AtomicBool::new(true));
        context.set_abort_flag(abort_flag);
        context.clear_abort_callback();

        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn synchronize_completes_without_panic() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        context.synchronize();
    }

    #[test]
    #[serial]
    fn detach_threadpool_completes_without_panic() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        context.detach_threadpool();
    }

    #[test]
    #[serial]
    fn mark_logits_initialized_records_token_index() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        context.mark_logits_initialized(0);

        assert_eq!(context.initialized_logits, vec![0]);
    }

    #[test]
    #[serial]
    fn print_memory_breakdown_completes_without_panic() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        context.print_memory_breakdown();
    }

    #[test]
    #[serial]
    fn get_logits_ith_returns_token_not_initialized_for_unknown_index() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.get_logits_ith(7);

        assert!(matches!(
            result,
            Err(crate::LogitsError::TokenNotInitialized(7))
        ));
    }

    #[test]
    #[serial]
    fn get_logits_ith_returns_token_index_exceeds_context_for_huge_index() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(64));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let huge_index = i32::try_from(context.n_ctx()).unwrap();
        context.mark_logits_initialized(huge_index);
        let result = context.get_logits_ith(huge_index);

        assert!(matches!(
            result,
            Err(crate::LogitsError::TokenIndexExceedsContext { .. })
        ));
    }
}
