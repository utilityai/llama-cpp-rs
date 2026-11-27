//! Safe wrapper around `llama_batch`.

use crate::context::LlamaContext;
use crate::ptr::Ptr;
use crate::token::LlamaToken;
use crate::{DecodeError, EncodeError};
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};
use std::marker::PhantomData;
use std::num::NonZeroI32;

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch<'a> {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    #[allow(clippy::doc_markdown)]
    /// The llama_cpp batch. always initialize by `llama_cpp_sys_2::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
    phantom: PhantomData<&'a [LlamaToken]>,
}

// SAFETY: The batch is POD, with getters using `&` and setters using `&mut`.
unsafe impl Send for LlamaBatch<'_> {}
// SAFETY: Same as above.
unsafe impl Sync for LlamaBatch<'_> {}

/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    /// Empty buffer is provided for [`LlamaBatch::get_one`]
    #[error("Empty buffer")]
    EmptyBuffer,
}

impl<'a> LlamaBatch<'a> {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
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
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// ``llama_batch_get_one``
    /// Return batch for single sequence of tokens
    ///
    /// NOTE: this is a helper function to facilitate transition to the new batch API
    ///
    /// # Errors
    /// If the provided token buffer is empty.
    ///
    /// # Panics
    /// If the number of tokens in ``tokens`` exceeds [`i32::MAX`].
    pub fn get_one(tokens: &'a [LlamaToken]) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }
        let batch = unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            llama_cpp_sys_2::llama_batch_get_one(
                ptr,
                tokens
                    .len()
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX"),
            )
        };
        let batch = Self {
            allocated: 0,
            llama_batch: batch,
            phantom: PhantomData,
        };
        Ok(batch)
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl<'a> Drop for LlamaBatch<'a> {
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

/// An extended batch of tokens or device embedding rows, bound to its processing context.
///
/// Processing consumes the batch and releases the context borrow, so output can then
/// be read from the context. All embedding entries must use device rows when any do;
/// state rows must likewise be set for all entries or none.
///
/// The context cannot be destroyed while a batch still uses its memory:
///
/// ```compile_fail
/// use llama_cpp_2::context::LlamaContext;
/// use llama_cpp_2::llama_batch::LlamaBatchExt;
/// fn cannot_drop_context(mut context: LlamaContext<'_>) {
///     let batch = LlamaBatchExt::new(&mut context);
///     drop(context);
///     drop(batch);
/// }
/// ```
#[derive(Debug)]
pub struct LlamaBatchExt<'context, 'model> {
    context: &'context mut LlamaContext<'model>,
    batch: Ptr<llama_cpp_sys_2::llama_batch_ext>,
}

/// An error code returned when an extended batch rejects an entry.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
#[error("extended batch rejected the entry with error code {0}")]
pub struct BatchExtAddError(pub i32);

/// An extended batch rejected a change to an existing entry.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
#[error("extended batch rejected the change to entry {index}")]
pub struct BatchExtEntryError {
    /// The batch index passed to the operation.
    pub index: i32,
}

impl<'context, 'model> LlamaBatchExt<'context, 'model> {
    /// Creates an empty batch using the context's model, limits, and memory.
    ///
    /// # Panics
    /// Panics if llama.cpp cannot allocate the batch.
    pub fn new(context: &'context mut LlamaContext<'model>) -> Self {
        let batch = unsafe { llama_cpp_sys_2::llama_batch_ext_init(context.context.as_mut_ptr()) };

        // The native batch retains the context's memory pointer, so hold its exclusive borrow until drop.
        let batch = Ptr::new(batch).expect("failed to allocate extended batch");
        Self { context, batch }
    }

    /// Clears the batch so its entries can be rebuilt before processing.
    pub fn clear(&mut self) {
        unsafe { llama_cpp_sys_2::llama_batch_ext_clear(self.batch.as_mut_ptr()) }
    }

    /// Adds a token and returns its batch index. Set its position before processing.
    ///
    /// # Errors
    ///
    /// Returns the C API error for a full batch, invalid sequence, or invalid token.
    pub fn add_token(
        &mut self,
        sequence_id: i32,
        token: LlamaToken,
    ) -> Result<i32, BatchExtAddError> {
        let index = unsafe {
            llama_cpp_sys_2::llama_batch_ext_add_token(
                self.batch.as_mut_ptr(),
                sequence_id,
                token.0,
            )
        };
        if index < 0 {
            Err(BatchExtAddError(index))
        } else {
            Ok(index)
        }
    }

    /// Adds a row from the paired context's layer input device buffer.
    /// Set its positions before processing. The context must support embedding rows.
    ///
    /// # Errors
    ///
    /// Returns the C API error for a full batch, invalid sequence, or invalid row.
    pub fn add_embedding_row(
        &mut self,
        sequence_id: i32,
        row: i32,
    ) -> Result<i32, BatchExtAddError> {
        let index = unsafe {
            llama_cpp_sys_2::llama_batch_ext_add_embd_row(self.batch.as_mut_ptr(), sequence_id, row)
        };
        if index < 0 {
            Err(BatchExtAddError(index))
        } else {
            Ok(index)
        }
    }

    /// Sets the state row read from the context configured with
    /// [`LlamaContext::set_context_state_source`].
    ///
    /// # Errors
    /// Returns an error for an invalid row, invalid index, or already-set entry.
    pub fn set_embedding_state_row(
        &mut self,
        index: i32,
        row: i32,
    ) -> Result<(), BatchExtEntryError> {
        let accepted = unsafe {
            llama_cpp_sys_2::llama_batch_ext_set_embd_state_row(self.batch.as_mut_ptr(), index, row)
        };
        accepted.then_some(()).ok_or(BatchExtEntryError { index })
    }

    /// Adds a sequence to an existing batch entry.
    ///
    /// # Errors
    /// Returns an error for invalid indices or sequences.
    pub fn add_sequence_id(
        &mut self,
        index: i32,
        sequence_id: i32,
    ) -> Result<(), BatchExtEntryError> {
        let accepted = unsafe {
            llama_cpp_sys_2::llama_batch_ext_add_seq(self.batch.as_mut_ptr(), index, sequence_id)
        };
        accepted.then_some(()).ok_or(BatchExtEntryError { index })
    }

    /// Sets the entry's positions in `[t, y, x, z]` order for M-RoPE embeddings.
    /// Text tokens and models without M-RoPE use only the first position.
    ///
    /// # Errors
    /// Returns an error for an invalid batch index.
    pub fn set_positions(
        &mut self,
        index: i32,
        positions: [i32; 4],
    ) -> Result<(), BatchExtEntryError> {
        // Native code reads up to four positions without receiving a length; the array guarantees that bound.
        let accepted = unsafe {
            llama_cpp_sys_2::llama_batch_ext_set_pos(
                self.batch.as_mut_ptr(),
                index,
                positions.as_ptr(),
            )
        };
        accepted.then_some(()).ok_or(BatchExtEntryError { index })
    }

    /// Sets whether an entry produces output logits.
    ///
    /// # Errors
    /// Returns an error for an invalid index.
    pub fn set_output_logits(
        &mut self,
        index: i32,
        enabled: bool,
    ) -> Result<(), BatchExtEntryError> {
        let accepted = unsafe {
            llama_cpp_sys_2::llama_batch_ext_set_output_logits(
                self.batch.as_mut_ptr(),
                index,
                enabled,
            )
        };
        accepted.then_some(()).ok_or(BatchExtEntryError { index })
    }

    /// Sets whether an entry produces output embeddings.
    ///
    /// # Errors
    /// Returns an error for an invalid index.
    pub fn set_output_embeddings(
        &mut self,
        index: i32,
        enabled: bool,
    ) -> Result<(), BatchExtEntryError> {
        let accepted = unsafe {
            llama_cpp_sys_2::llama_batch_ext_set_output_embd(
                self.batch.as_mut_ptr(),
                index,
                enabled,
            )
        };
        accepted.then_some(()).ok_or(BatchExtEntryError { index })
    }

    /// Decodes the batch and releases its context borrow.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`LlamaContext::decode`].
    pub fn decode(mut self) -> Result<(), DecodeError> {
        let result = unsafe {
            llama_cpp_sys_2::llama_process(
                self.context.context.as_mut_ptr(),
                llama_cpp_sys_2::LLAMA_PROCESS_TYPE_DECODE,
                self.batch.as_mut_ptr(),
            )
        };
        match NonZeroI32::new(result) {
            None => Ok(()),
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch and releases its context borrow.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`LlamaContext::encode`].
    pub fn encode(mut self) -> Result<(), EncodeError> {
        let result = unsafe {
            llama_cpp_sys_2::llama_process(
                self.context.context.as_mut_ptr(),
                llama_cpp_sys_2::LLAMA_PROCESS_TYPE_ENCODE,
                self.batch.as_mut_ptr(),
            )
        };
        match NonZeroI32::new(result) {
            None => Ok(()),
            Some(error) => Err(EncodeError::from(error)),
        }
    }
}

impl Drop for LlamaBatchExt<'_, '_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_batch_ext_free(self.batch.as_mut_ptr()) }
    }
}
