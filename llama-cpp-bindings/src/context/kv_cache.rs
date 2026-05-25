use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};
use std::os::raw::c_char;
use std::ptr;

use crate::context::LlamaContext;
use crate::error::{KvCacheSeqAddError, KvCacheSeqDivError};
use crate::ffi_error_reader::read_and_free_cpp_error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheConversionError {
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
}

impl LlamaContext<'_> {
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, 0, size) }
    }

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

    pub fn clear_kv_cache(&mut self) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        let clear_data_buffers = true;
        unsafe { llama_cpp_bindings_sys::llama_memory_clear(mem, clear_data_buffers) }
    }

    pub fn kv_cache_seq_keep(&mut self, seq_id: i32) {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_keep(mem, seq_id) }
    }

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
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE => {
                Err(KvCacheSeqAddError::IncompatibleRopeType)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM => {
                Err(KvCacheSeqAddError::MemoryHandleUnavailable)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED => {
                Err(KvCacheSeqAddError::NotEnoughMemory)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { read_and_free_cpp_error(out_error) };
                Err(KvCacheSeqAddError::Reported { message })
            }
            other => unreachable!("llama_rs_memory_seq_add returned unrecognized status {other}"),
        }
    }

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
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE => {
                Err(KvCacheSeqDivError::IncompatibleRopeType)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM => {
                Err(KvCacheSeqDivError::MemoryHandleUnavailable)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED => {
                Err(KvCacheSeqDivError::NotEnoughMemory)
            }
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { read_and_free_cpp_error(out_error) };
                Err(KvCacheSeqDivError::Reported { message })
            }
            other => unreachable!("llama_rs_memory_seq_div returned unrecognized status {other}"),
        }
    }

    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max(self.context.as_ptr(), seq_id)
        }
    }
}
