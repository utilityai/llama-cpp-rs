//! Safe wrapper around `llama_batch`.

use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initilized
    allocated: usize,
    /// The logits that are initilized. Used by [`LlamaContext`] to ensure that only initilized logits are accessed.
    pub(crate) initialized_logits: Vec<i32>,
    /// The llama_cpp batch. always initilize by `llama_cpp_sys_2::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
}

impl LlamaBatch {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// Set the last token in the batch to [value]. If [value] is true, the token will be initilized
    /// after a decode and can be read from. If [value] is false, the token will not be initilized (this is the default).
    ///
    /// # Panics
    ///
    /// Panics if there are no tokens in the batch.
    #[deprecated(
        note = "not compatible with multiple sequences. prefer setting logits while adding tokens"
    )]
    pub fn set_last_logit(&mut self, value: bool) {
        let last_index = self.llama_batch.n_tokens - 1;
        let last_index_usize =
            usize::try_from(last_index).expect("cannot fit n_tokens - 1 into a usize");

        if value {
            self.initialized_logits.push(last_index);
        } else {
            self.initialized_logits.retain(|&x| x != last_index);
        }

        let value = i8::from(value);
        unsafe {
            let last: *mut i8 = self.llama_batch.logits.add(last_index_usize);
            *last = value;
        }
    }

    /// add a token to the batch for sequences [`seq_ids`] at position [pos]. If [logits] is true, the
    /// token will be initilized and can be read from after the next decode.
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a usize
    /// - [`seq_ids.len()`] does not fit into a [`llama_seq_id`]
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) {
        assert!(self.allocated > (usize::try_from(self.n_tokens() + 1).expect("self.n_tokens does not fit into a usize")), "there are only {} tokens allocated for the batch, but {} tokens in the batch when you tried to add one", self.allocated, self.n_tokens());
        unsafe {
            // batch.token   [batch.n_tokens] = id;
            let offset = self.llama_batch.n_tokens;
            let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
            *self.llama_batch.token.add(offset_usize) = id;
            // batch.pos     [batch.n_tokens] = pos,
            *self.llama_batch.pos.add(offset_usize) = pos;
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            *self.llama_batch.n_seq_id.add(offset_usize) = llama_seq_id::try_from(seq_ids.len())
                .expect("cannot fit seq_ids.len() into a llama_seq_id");
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                *tmp.add(i) = *seq_id;
            }
            // batch.logits  [batch.n_tokens] = logits;
            *self.llama_batch.logits.add(offset_usize) = i8::from(logits);

            if logits {
                self.initialized_logits.push(offset);
            } else {
                self.initialized_logits.retain(|l| l != &offset);
            }

            // batch.n_tokens++;
            self.llama_batch.n_tokens += 1;
        }
    }
    /// Create a new `LlamaBatch` that cab contain up to `n_tokens` tokens.
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is greater than `i32::MAX`.
    #[must_use]
    pub fn new(n_tokens: usize, embd: i32, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let batch = unsafe { llama_batch_init(n_tokens_i32, embd, n_seq_max) };

        LlamaBatch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
        }
    }

    /// add a prompt to the batch at sequence id 0
    #[deprecated(note = "not compatible with multiple sequences. use `add_prompt_seq` instead")]
    pub fn add_prompt(&mut self, prompt: &[LlamaToken]) {
        self.add_prompt_seq(prompt, &[0]);
    }

    /// add a prompt to the batch at the given sequence ids. This must be the initial prompt as it
    /// will be added to the batch starting at position 0.
    pub fn add_prompt_seq(&mut self, prompt: &[LlamaToken], seq_ids: &[i32]) {
        for (i, &token) in (0_i32..).zip(prompt) {
            self.add(token, i, seq_ids, false);
        }
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl Drop for LlamaBatch {
    /// Drops the `LlamaBatch`.
    ///
    /// ```
    /// # use llama_cpp::llama_batch::LlamaBatch;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let batch = LlamaBatch::new_from_prompt(&[]);
    /// // frees the memory associated with the batch. (allocated by llama.cpp)
    /// drop(batch);
    /// # Ok(())
    /// # }
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.llama_batch);
        }
    }
}
