//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[allow(clippy::module_name_repetitions)]
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
        unsafe { llama_cpp_sys_2::llama_kv_self_seq_cp(self.context.as_ptr(), src, dest, 0, size) }
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
        unsafe {
            llama_cpp_sys_2::llama_kv_self_seq_cp(self.context.as_ptr(), src, dest, p0, p1);
        }
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
        Ok(unsafe { llama_cpp_sys_2::llama_kv_self_seq_rm(self.context.as_ptr(), src, p0, p1) })
    }

    /// Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    #[must_use]
    pub fn get_kv_cache_used_cells(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_kv_self_used_cells(self.context.as_ptr()) }
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_self_clear(self.context.as_ptr()) }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        unsafe { llama_cpp_sys_2::llama_kv_self_seq_keep(self.context.as_ptr(), seq_id) }
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
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
        unsafe {
            llama_cpp_sys_2::llama_kv_self_seq_add(self.context.as_ptr(), seq_id, p0, p1, delta);
        }
        Ok(())
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
        unsafe { llama_cpp_sys_2::llama_kv_self_seq_div(self.context.as_ptr(), seq_id, p0, p1, d) }
        Ok(())
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe { llama_cpp_sys_2::llama_kv_self_seq_pos_max(self.context.as_ptr(), seq_id) }
    }

    /// Defragment the KV cache
    /// This will be applied:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    pub fn kv_cache_defrag(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_self_defrag(self.context.as_ptr()) }
    }

    /// Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    pub fn kv_cache_update(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_self_update(self.context.as_ptr()) }
    }
}
