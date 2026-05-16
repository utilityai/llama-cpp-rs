//! utilities for working with the kv cache

use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};
use std::os::raw::c_char;
use std::ptr;

use crate::context::LlamaContext;
use crate::error::{KvCacheSeqAddError, KvCacheSeqDivError};
use crate::ffi_error_reader::read_and_free_cpp_error;

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
    /// If either position exceeds [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheSeqAddError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqAddError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqAddError::P1TooLarge)?;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_add(
                self.context.as_ptr(),
                seq_id,
                p0,
                p1,
                delta,
                &raw mut out_error,
            )
        };
        match status {
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK => Ok(()),
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG => {
                Err(KvCacheSeqAddError::NullContextArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE => {
                Err(KvCacheSeqAddError::IncompatibleRopeType)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM => {
                Err(KvCacheSeqAddError::NullMem)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED => {
                Err(KvCacheSeqAddError::ErrorStringAllocationFailed)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { read_and_free_cpp_error(out_error) };
                Err(KvCacheSeqAddError::VendoredThrewCxxException { message })
            }
            other => unreachable!("llama_rs_memory_seq_add returned unrecognized status {other}"),
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
    /// If either position exceeds [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheSeqDivError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqDivError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqDivError::P1TooLarge)?;
        let d = c_int::from(d.get());
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_div(
                self.context.as_ptr(),
                seq_id,
                p0,
                p1,
                d,
                &raw mut out_error,
            )
        };
        match status {
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK => Ok(()),
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG => {
                Err(KvCacheSeqDivError::NullContextArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE => {
                Err(KvCacheSeqDivError::IncompatibleRopeType)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM => {
                Err(KvCacheSeqDivError::NullMem)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED => {
                Err(KvCacheSeqDivError::ErrorStringAllocationFailed)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { read_and_free_cpp_error(out_error) };
                Err(KvCacheSeqDivError::VendoredThrewCxxException { message })
            }
            other => unreachable!("llama_rs_memory_seq_div returned unrecognized status {other}"),
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
