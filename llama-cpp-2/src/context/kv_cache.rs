//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::NonZeroU8;

impl LlamaContext {
    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        unsafe { llama_cpp_sys_2::llama_kv_cache_seq_cp(self.context.as_ptr(), src, dest, 0, size) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    pub fn copy_kv_cache_seq(&mut self, src: i32, dest: i32, p0: Option<u16>, p1: Option<u16>) {
        let p0 = p0.map_or(-1, i32::from);
        let p1 = p1.map_or(-1, i32::from);
        unsafe {
            llama_cpp_sys_2::llama_kv_cache_seq_cp(self.context.as_ptr(), src, dest, p0, p1);
        }
    }

    /// Clear the kv cache for the given sequence.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to clear the cache for.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    pub fn clear_kv_cache_seq(&mut self, src: i32, p0: Option<u16>, p1: Option<u16>) {
        let p0 = p0.map_or(-1, i32::from);
        let p1 = p1.map_or(-1, i32::from);
        unsafe {
            llama_cpp_sys_2::llama_kv_cache_seq_rm(self.context.as_ptr(), src, p0, p1);
        }
    }

    /// Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    #[must_use]
    pub fn get_kv_cache_used_cells(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_get_kv_cache_used_cells(self.context.as_ptr()) }
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_cache_clear(self.context.as_ptr()) }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        unsafe { llama_cpp_sys_2::llama_kv_cache_seq_keep(self.context.as_ptr(), seq_id) }
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    pub fn kv_cache_seq_add(&mut self, seq_id: i32, p0: Option<u16>, p1: Option<u16>, delta: i32) {
        let p0 = p0.map_or(-1, i32::from);
        let p1 = p1.map_or(-1, i32::from);
        unsafe {
            llama_cpp_sys_2::llama_kv_cache_seq_add(self.context.as_ptr(), seq_id, p0, p1, delta);
        }
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u16>,
        p1: Option<u16>,
        d: NonZeroU8,
    ) {
        let p0 = p0.map_or(-1, i32::from);
        let p1 = p1.map_or(-1, i32::from);
        let d = c_int::from(d.get());
        unsafe { llama_cpp_sys_2::llama_kv_cache_seq_div(self.context.as_ptr(), seq_id, p0, p1, d) }
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe { llama_cpp_sys_2::llama_kv_cache_seq_pos_max(self.context.as_ptr(), seq_id) }
    }

    /// Defragment the KV cache
    /// This will be applied:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    pub fn kv_cache_defrag(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_cache_defrag(self.context.as_ptr()) }
    }

    /// Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    pub fn kv_cache_update(&mut self) {
        unsafe { llama_cpp_sys_2::llama_kv_cache_update(self.context.as_ptr()) }
    }

    /// Returns the number of tokens in the KV cache (slow, use only for debug)
    /// If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    #[must_use]
    pub fn get_kv_cache_token_count(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_get_kv_cache_token_count(self.context.as_ptr()) }
    }

    /// Create an empty KV cache view. (use only for debugging purposes)
    ///
    /// # Parameters
    ///
    /// * `n_max_seq` - Maximum number of sequences that can exist in a cell. It's not an error
    ///                 if there are more sequences in a cell than this value, however they will
    ///                 not be visible in the view `cells_sequences`.
    #[must_use]
    pub fn new_kv_cache_view(&self, n_max_seq: i32) -> KVCacheView {
        let view =
            unsafe { llama_cpp_sys_2::llama_kv_cache_view_init(self.context.as_ptr(), n_max_seq) };
        KVCacheView { view, ctx: self }
    }
}

/// Information associated with an individual cell in the KV cache view.
#[derive(Debug)]
pub struct KVCacheViewCell {
    /// The position for this cell. Takes KV cache shifts into account.
    /// May be negative if the cell is not populated.
    pub pos: llama_cpp_sys_2::llama_pos,
}

/// An updateable view of the KV cache. (use only for debugging purposes)
#[derive(Debug)]
pub struct KVCacheView<'a> {
    ctx: &'a LlamaContext,
    view: llama_cpp_sys_2::llama_kv_cache_view,
}

impl<'a> KVCacheView<'a> {
    /// Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    pub fn update(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_kv_cache_view_update(self.ctx.context.as_ptr(), &mut self.view);
        }
    }

    /// Number of KV cache cells. This will be the same as the context size.
    #[must_use]
    pub fn n_cells(&self) -> i32 {
        self.view.n_cells
    }

    /// Number of tokens in the cache. For example, if there are two populated
    /// cells, the first with 1 sequence id in it and the second with 2 sequence
    /// ids then you'll have 3 tokens.
    #[must_use]
    pub fn token_count(&self) -> i32 {
        self.view.token_count
    }

    /// Number of populated cache cells.
    #[must_use]
    pub fn used_cells(&self) -> i32 {
        self.view.used_cells
    }

    /// Maximum contiguous empty slots in the cache.
    #[must_use]
    pub fn max_contiguous(&self) -> i32 {
        self.view.max_contiguous
    }

    /// Index to the start of the `max_contiguous` slot range. Can be negative
    /// when cache is full.
    #[must_use]
    pub fn max_contiguous_idx(&self) -> i32 {
        self.view.max_contiguous_idx
    }

    /// Information for individual cells.
    ///
    /// # Panics
    ///
    /// - if `n_cells` does not fit into usize.
    pub fn cells(&self) -> impl Iterator<Item = KVCacheViewCell> {
        unsafe {
            std::slice::from_raw_parts(
                self.view.cells,
                usize::try_from(self.view.n_cells).expect("failed to fit n_cells into usize"),
            )
        }
        .iter()
        .map(|&cell| KVCacheViewCell { pos: cell.pos })
    }

    /// The sequences for each cell. There will be `n_max_seq` items per cell.
    ///
    /// # Panics
    ///
    /// - if `n_cells * n_max_seq` does not fit into usize.
    /// - if `n_max_seq` does not fit into usize.
    pub fn cells_sequences(&self) -> impl Iterator<Item = &[llama_cpp_sys_2::llama_seq_id]> {
        unsafe {
            std::slice::from_raw_parts(
                self.view.cells_sequences,
                usize::try_from(self.view.n_cells * self.view.n_seq_max)
                    .expect("failed to fit n_cells * n_max_seq into usize"),
            )
        }
        .chunks(usize::try_from(self.view.n_seq_max).expect("failed to fit n_max_seq into usize"))
    }
}

impl<'a> Drop for KVCacheView<'a> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_kv_cache_view_free(&mut self.view);
        }
    }
}
