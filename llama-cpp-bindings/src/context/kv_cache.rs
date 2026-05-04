//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheConversionError {
    /// Sequence id conversion to i32 failed
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    /// Position 0 conversion to i32 failed
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    /// Position 1 conversion to i32 failed
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
    /// The operation is not supported by the current model/context configuration.
    #[error("operation not supported by this model: {0}")]
    UnsupportedOperation(String),
}

impl LlamaContext<'_> {
    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, 0, size) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, p0, p1) };
        Ok(())
    }

    /// Clear the kv cache for the given sequence within the specified range `[p0, p1)`
    /// Returns `false` only when partial sequence removals fail. Full sequence removals always succeed.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful. If the sequence id or
    /// either position exceeds the maximum i32 value, no removal is attempted and an `Err` is returned.
    ///
    /// # Parameters
    /// * `src` - The sequence id to clear the cache for. If `None`, matches all sequences
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    ///
    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`].
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        Ok(unsafe { llama_cpp_bindings_sys::llama_memory_seq_rm(mem, src, p0, p1) })
    }

    /// Clear the KV cache, including both metadata and the underlying data buffers.
    pub fn clear_kv_cache(&mut self) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        let clear_data_buffers = true;
        unsafe { llama_cpp_bindings_sys::llama_memory_clear(mem, clear_data_buffers) }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn kv_cache_seq_keep(&mut self, seq_id: i32) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_keep(mem, seq_id) }
    }

    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_add(
                self.context.as_ptr(),
                seq_id,
                p0,
                p1,
                delta,
            )
        };

        if crate::status_is_ok(status) {
            Ok(())
        } else {
            Err(KvCacheConversionError::UnsupportedOperation(format!(
                "kv_cache_seq_add failed (status {})",
                crate::status_to_i32(status)
            )))
        }
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let d = c_int::from(d.get());
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_div(
                self.context.as_ptr(),
                seq_id,
                p0,
                p1,
                d,
            )
        };

        if crate::status_is_ok(status) {
            Ok(())
        } else {
            Err(KvCacheConversionError::UnsupportedOperation(format!(
                "kv_cache_seq_div failed (status {})",
                crate::status_to_i32(status)
            )))
        }
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max(self.context.as_ptr(), seq_id)
        }
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use std::num::NonZeroU32;

    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::llama_batch::LlamaBatch;
    use crate::model::AddBos;
    use crate::test_model;

    #[test]
    #[serial]
    fn clear_kv_cache_resets_positions() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        context.clear_kv_cache();
        assert_eq!(context.kv_cache_seq_pos_max(0), -1);
    }

    #[test]
    #[serial]
    fn kv_cache_seq_pos_max_is_non_negative_after_decode() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        assert!(context.kv_cache_seq_pos_max(0) >= 0);
    }

    #[test]
    #[serial]
    fn clear_kv_cache_seq_with_range() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.clear_kv_cache_seq(Some(0), Some(0), Some(1));
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn copy_kv_cache_seq_succeeds() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.copy_kv_cache_seq(0, 1, None, None);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn copy_cache_executes_without_crash() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let pos_max = context.kv_cache_seq_pos_max(0);
        context.copy_cache(0, 1, pos_max + 1);
    }

    #[test]
    #[serial]
    fn kv_cache_seq_add_returns_error_for_mrope_model() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.kv_cache_seq_add(0, Some(0), None, 1);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn kv_cache_seq_div_returns_error_for_mrope_model() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let divisor = std::num::NonZeroU8::new(2).unwrap();
        let result = context.kv_cache_seq_div(0, Some(0), None, divisor);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn kv_cache_seq_keep_retains_specified_sequence() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        context.kv_cache_seq_keep(0);

        assert!(context.kv_cache_seq_pos_max(0) >= 0);
    }

    #[test]
    #[serial]
    fn copy_kv_cache_seq_with_explicit_range() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.copy_kv_cache_seq(0, 2, Some(0), Some(1));

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn kv_cache_seq_add_succeeds_on_embedding_model() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.kv_cache_seq_add(0, Some(0), None, 1);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn kv_cache_seq_div_succeeds_on_embedding_model() {
        let (backend, model) = test_model::load_default_embedding_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let divisor = std::num::NonZeroU8::new(2).unwrap();
        let result = context.kv_cache_seq_div(0, Some(0), None, divisor);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn kv_cache_seq_pos_max_returns_negative_one_for_unused_seq() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.kv_cache_seq_pos_max(999);

        assert_eq!(result, -1);
    }

    #[test]
    #[serial]
    fn copy_kv_cache_seq_rejects_p0_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.copy_kv_cache_seq(0, 1, Some(u32::MAX), None);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P0TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn copy_kv_cache_seq_rejects_p1_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.copy_kv_cache_seq(0, 1, Some(0), Some(u32::MAX));

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P1TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn clear_kv_cache_seq_rejects_src_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.clear_kv_cache_seq(Some(u32::MAX), None, None);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::SeqIdTooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn clear_kv_cache_seq_rejects_p0_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.clear_kv_cache_seq(Some(0), Some(u32::MAX), None);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P0TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn clear_kv_cache_seq_rejects_p1_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.clear_kv_cache_seq(Some(0), Some(0), Some(u32::MAX));

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P1TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn kv_cache_seq_add_rejects_p0_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.kv_cache_seq_add(0, Some(u32::MAX), None, 1);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P0TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn kv_cache_seq_add_rejects_p1_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.kv_cache_seq_add(0, Some(0), Some(u32::MAX), 1);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P1TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn kv_cache_seq_div_rejects_p0_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let divisor = std::num::NonZeroU8::new(2).unwrap();
        let result = context.kv_cache_seq_div(0, Some(u32::MAX), None, divisor);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P0TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }

    #[test]
    #[serial]
    fn kv_cache_seq_div_rejects_p1_exceeding_i32_max() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let divisor = std::num::NonZeroU8::new(2).unwrap();
        let result = context.kv_cache_seq_div(0, Some(0), Some(u32::MAX), divisor);

        assert_eq!(
            result.unwrap_err(),
            super::KvCacheConversionError::P1TooLarge(i32::try_from(u32::MAX).unwrap_err()),
        );
    }
}
