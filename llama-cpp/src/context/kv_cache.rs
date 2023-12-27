//! utilities for working with the kv cache

use crate::context::LlamaContext;

impl LlamaContext<'_> {
    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        unsafe { llama_cpp_sys::llama_kv_cache_seq_cp(self.context.as_ptr(), src, dest, 0, size) }
    }

    /// Clear the kv cache for the given sequence.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to clear the cache for.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to [p1].
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from [p0].
    pub fn clear_kv_cache_seq(&mut self, src: i32, p0: Option<u16>, p1: Option<u16>) {
        unsafe {
            llama_cpp_sys::llama_kv_cache_seq_rm(
                self.context.as_ptr(),
                src,
                p0.map_or(-1, i32::from),
                p1.map_or(-1, i32::from),
            );
        }
    }
}
