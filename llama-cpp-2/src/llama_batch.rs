//! Safe wrapper around `llama_batch`.

use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    /// The logits that are initialized. Used by [`LlamaContext`] to ensure that only initialized logits are accessed.
    pub(crate) initialized_logits: Vec<i32>,
    /// The llama_cpp batch. always initialize by `llama_cpp_sys_2::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
}

/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
}

impl LlamaBatch {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// add a token to the batch for sequences `seq_ids` at position `pos`. If `logits` is true, the
    /// token will be initialized and can be read from after the next decode.
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a usize
    /// - [`seq_ids.len()`] does not fit into a [`llama_seq_id`]
    ///
    /// # Errors
    ///
    /// returns a error if there is insufficient space in the buffer
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token   [batch.n_tokens] = id;
            self.llama_batch.token.add(offset_usize).write(id);
            // batch.pos     [batch.n_tokens] = pos,
            self.llama_batch.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            // batch.logits  [batch.n_tokens] = logits;
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        // batch.n_tokens++;
        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    /// Add a sequence of tokens to the batch for the given sequence id. If `logits_all` is true, the
    /// tokens will be initialized and can be read from after the next decode.
    ///
    /// Either way the last token in the sequence will have its logits set to `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a [`usize`]
    /// - [`n_tokens - 1`] does not fit into a [`llama_pos`]
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let n_tokens_0 =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        let n_tokens = tokens.len();

        if self.allocated < n_tokens_0 + n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let last_index = llama_pos::try_from(n_tokens.saturating_sub(1))
            .expect("cannot fit n_tokens into a llama_pos");
        for (i, token) in (0..).zip(tokens.iter()) {
            self.add(*token, i, &[seq_id], logits_all || i == last_index)?;
        }

        Ok(())
    }

    /// Create a new `LlamaBatch` that can contain up to `n_tokens` tokens.
    ///
    /// # Arguments
    ///
    /// - `n_tokens`: the maximum number of tokens that can be added to the batch
    /// - `n_seq_max`: the maximum number of sequences that can be added to the batch (generally 1 unless you know what you are doing)
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is greater than `i32::MAX`.
    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let batch = unsafe { llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        LlamaBatch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
        }
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }

    /// llama_batch_get_one
    /// Return batch for single sequence of tokens starting at pos_0
    ///
    /// NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    ///
    pub fn get_one(tokens: &[LlamaToken], pos_0: llama_pos, seq_id: llama_seq_id) -> Self {
        unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            let batch =
                llama_cpp_sys_2::llama_batch_get_one(ptr, tokens.len() as i32, pos_0, seq_id);

            LlamaBatch {
                allocated: 0,
                initialized_logits: vec![],
                llama_batch: batch,
            }
        }
    }
}

impl Drop for LlamaBatch {
    /// Drops the `LlamaBatch`.
    ///
    /// ```
    /// # use llama_cpp_2::llama_batch::LlamaBatch;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let batch = LlamaBatch::new(512, 1);
    /// // frees the memory associated with the batch. (allocated by llama.cpp)
    /// drop(batch);
    /// # Ok(())
    /// # }
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_batch_free(self.llama_batch);
            }
        }
    }
}
