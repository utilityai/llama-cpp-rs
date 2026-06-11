use std::marker::PhantomData;

use llama_cpp_bindings_sys::{
    llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id,
};

use crate::batch_add_error::BatchAddError;
use crate::sampled_token::SampledToken;
use crate::token::LlamaToken;

fn checked_n_tokens_plus_one_as_usize(n_tokens: i32) -> Result<usize, BatchAddError> {
    let incremented = n_tokens.checked_add(1).ok_or_else(|| {
        BatchAddError::IntegerOverflow(format!("n_tokens + 1 overflows i32: {n_tokens}"))
    })?;

    usize::try_from(incremented).map_err(|convert_error| {
        BatchAddError::IntegerOverflow(format!("cannot fit n_tokens into a usize: {convert_error}"))
    })
}

fn checked_i32_as_usize(value: i32, description: &str) -> Result<usize, BatchAddError> {
    usize::try_from(value).map_err(|convert_error| {
        BatchAddError::IntegerOverflow(format!(
            "cannot fit {description} into a usize: {convert_error}"
        ))
    })
}

fn checked_usize_as_llama_seq_id(
    value: usize,
    description: &str,
) -> Result<llama_seq_id, BatchAddError> {
    llama_seq_id::try_from(value).map_err(|convert_error| {
        BatchAddError::IntegerOverflow(format!(
            "cannot fit {description} into a llama_seq_id: {convert_error}"
        ))
    })
}

fn checked_usize_as_i32(value: usize, description: &str) -> Result<i32, BatchAddError> {
    i32::try_from(value).map_err(|convert_error| {
        BatchAddError::IntegerOverflow(format!(
            "cannot fit {description} into a i32: {convert_error}"
        ))
    })
}

fn checked_usize_as_llama_pos(value: usize, description: &str) -> Result<llama_pos, BatchAddError> {
    llama_pos::try_from(value).map_err(|convert_error| {
        BatchAddError::IntegerOverflow(format!(
            "cannot fit {description} into a llama_pos: {convert_error}"
        ))
    })
}

#[derive(Debug)]
pub struct LlamaBatch<'tokens> {
    allocated: usize,
    pub initialized_logits: Vec<i32>,
    pub llama_batch: llama_batch,
    phantom: PhantomData<&'tokens [LlamaToken]>,
}

impl<'tokens> LlamaBatch<'tokens> {
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer or if integer conversions fail.
    pub fn add(
        &mut self,
        sampled_token: &SampledToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        let (SampledToken::Content(LlamaToken(id))
        | SampledToken::Reasoning(LlamaToken(id))
        | SampledToken::ToolCall(LlamaToken(id))
        | SampledToken::Undeterminable(LlamaToken(id))) = *sampled_token;
        let required = checked_n_tokens_plus_one_as_usize(self.n_tokens())?;

        if self.allocated < required {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let offset = self.llama_batch.n_tokens;
        let offset_usize = checked_i32_as_usize(offset, "n_tokens")?;
        let n_seq_id = checked_usize_as_llama_seq_id(seq_ids.len(), "seq_ids.len()")?;

        unsafe {
            self.llama_batch.token.add(offset_usize).write(id);
            self.llama_batch.pos.add(offset_usize).write(pos);
            self.llama_batch.n_seq_id.add(offset_usize).write(n_seq_id);
            for (seq_index, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(seq_index).write(*seq_id);
            }
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        }

        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer or if integer conversions fail.
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let last_index = checked_usize_as_llama_pos(tokens.len().saturating_sub(1), "n_tokens")?;

        for (position, token) in (0..).zip(tokens.iter()) {
            self.add(
                &SampledToken::Content(*token),
                position,
                &[seq_id],
                logits_all || position == last_index,
            )?;
        }

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if `n_tokens` exceeds `i32::MAX`.
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Result<Self, BatchAddError> {
        let n_tokens_i32 = checked_usize_as_i32(n_tokens, "n_tokens")?;
        let batch = unsafe { llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        Ok(LlamaBatch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        })
    }

    /// # Errors
    ///
    /// Returns an error if the provided token buffer is empty or if integer conversions fail.
    pub fn get_one(tokens: &'tokens [LlamaToken]) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }

        let token_count = checked_usize_as_i32(tokens.len(), "token count")?;

        let batch = unsafe {
            let ptr = tokens.as_ptr().cast::<i32>().cast_mut();
            llama_cpp_bindings_sys::llama_batch_get_one(ptr, token_count)
        };

        let last_token_index = checked_usize_as_i32(tokens.len() - 1, "last token index")?;

        Ok(Self {
            allocated: 0,
            initialized_logits: vec![last_token_index],
            llama_batch: batch,
            phantom: PhantomData,
        })
    }

    #[must_use]
    pub const fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl Drop for LlamaBatch<'_> {
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_batch_free(self.llama_batch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    use super::{
        BatchAddError, LlamaBatch, checked_i32_as_usize, checked_n_tokens_plus_one_as_usize,
        checked_usize_as_i32, checked_usize_as_llama_pos, checked_usize_as_llama_seq_id,
    };

    #[test]
    fn new_creates_empty_batch() {
        let batch = LlamaBatch::new(16, 1).unwrap();

        assert_eq!(batch.n_tokens(), 0);
        assert!(batch.initialized_logits.is_empty());
    }

    #[test]
    fn clear_resets_batch() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], true)
            .unwrap();
        assert_eq!(batch.n_tokens(), 1);

        batch.clear();

        assert_eq!(batch.n_tokens(), 0);
        assert!(batch.initialized_logits.is_empty());
    }

    #[test]
    fn add_increments_token_count() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();

        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false)
            .unwrap();
        assert_eq!(batch.n_tokens(), 1);

        batch
            .add(&SampledToken::Content(LlamaToken::new(2)), 1, &[0], false)
            .unwrap();
        assert_eq!(batch.n_tokens(), 2);
    }

    #[test]
    fn add_tracks_logits() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();

        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false)
            .unwrap();
        assert!(batch.initialized_logits.is_empty());

        batch
            .add(&SampledToken::Content(LlamaToken::new(2)), 1, &[0], true)
            .unwrap();
        assert_eq!(batch.initialized_logits, vec![1]);
    }

    #[test]
    fn add_returns_insufficient_space_when_full() {
        let mut batch = LlamaBatch::new(1, 1).unwrap();
        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false)
            .unwrap();

        let result = batch.add(&SampledToken::Content(LlamaToken::new(2)), 1, &[0], false);

        assert_eq!(result, Err(BatchAddError::InsufficientSpace(1)));
    }

    #[test]
    fn add_accepts_reasoning_sampled_token_variant() {
        let mut batch = LlamaBatch::new(4, 1).unwrap();

        batch
            .add(&SampledToken::Reasoning(LlamaToken::new(11)), 0, &[0], true)
            .unwrap();

        assert_eq!(batch.n_tokens(), 1);
    }

    #[test]
    fn add_accepts_tool_call_sampled_token_variant() {
        let mut batch = LlamaBatch::new(4, 1).unwrap();

        batch
            .add(&SampledToken::ToolCall(LlamaToken::new(22)), 0, &[0], true)
            .unwrap();

        assert_eq!(batch.n_tokens(), 1);
    }

    #[test]
    fn add_accepts_undeterminable_sampled_token_variant() {
        let mut batch = LlamaBatch::new(4, 1).unwrap();

        batch
            .add(
                &SampledToken::Undeterminable(LlamaToken::new(33)),
                0,
                &[0],
                false,
            )
            .unwrap();

        assert_eq!(batch.n_tokens(), 1);
    }

    #[test]
    fn add_sequence_adds_all_tokens() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        let tokens = vec![
            LlamaToken::new(10),
            LlamaToken::new(20),
            LlamaToken::new(30),
        ];

        batch.add_sequence(&tokens, 0, false).unwrap();

        assert_eq!(batch.n_tokens(), 3);
    }

    #[test]
    fn add_sequence_sets_logits_on_last_token() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        let tokens = vec![
            LlamaToken::new(10),
            LlamaToken::new(20),
            LlamaToken::new(30),
        ];

        batch.add_sequence(&tokens, 0, false).unwrap();

        assert_eq!(batch.initialized_logits, vec![2]);
    }

    #[test]
    fn add_sequence_insufficient_space() {
        let mut batch = LlamaBatch::new(2, 1).unwrap();
        let tokens = vec![
            LlamaToken::new(10),
            LlamaToken::new(20),
            LlamaToken::new(30),
        ];

        let result = batch.add_sequence(&tokens, 0, false);

        assert!(result.is_err());
    }

    #[test]
    fn add_sequence_fails_mid_loop_when_batch_fills() {
        let mut batch = LlamaBatch::new(2, 1).unwrap();
        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false)
            .unwrap();

        let tokens = vec![LlamaToken::new(10), LlamaToken::new(20)];
        let result = batch.add_sequence(&tokens, 0, false);

        assert!(result.is_err());
    }

    #[test]
    fn get_one_with_valid_tokens() {
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2)];
        let batch = LlamaBatch::get_one(&tokens).expect("test: get_one should succeed");

        assert_eq!(batch.n_tokens(), 2);
        assert_eq!(batch.initialized_logits, vec![1]);
    }

    #[test]
    fn get_one_empty_slice_returns_error() {
        let tokens: Vec<LlamaToken> = vec![];
        let result = LlamaBatch::get_one(&tokens);

        assert_eq!(result.unwrap_err(), BatchAddError::EmptyBuffer);
    }

    #[test]
    fn get_one_single_token() {
        let tokens = vec![LlamaToken::new(42)];
        let batch = LlamaBatch::get_one(&tokens).expect("test: get_one should succeed");

        assert_eq!(batch.n_tokens(), 1);
        assert_eq!(batch.initialized_logits, vec![0]);
    }

    #[test]
    fn add_with_logits_false_retains_only_previous_logits() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();

        batch
            .add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], true)
            .unwrap();
        assert_eq!(batch.initialized_logits, vec![0]);

        batch
            .add(&SampledToken::Content(LlamaToken::new(2)), 0, &[0], false)
            .unwrap();
        assert_eq!(batch.initialized_logits, vec![0]);
    }

    #[test]
    fn add_sequence_with_logits_all_marks_every_token() -> Result<(), BatchAddError> {
        let mut batch = LlamaBatch::new(16, 1)?;
        let tokens = vec![
            LlamaToken::new(10),
            LlamaToken::new(20),
            LlamaToken::new(30),
        ];

        batch.add_sequence(&tokens, 0, true)?;

        assert_eq!(batch.n_tokens(), 3);
        assert_eq!(batch.initialized_logits, vec![0, 1, 2]);

        Ok(())
    }

    #[test]
    fn add_with_multiple_seq_ids() -> Result<(), BatchAddError> {
        let mut batch = LlamaBatch::new(16, 4)?;

        batch.add(
            &SampledToken::Content(LlamaToken::new(1)),
            0,
            &[0, 1, 2],
            true,
        )?;

        assert_eq!(batch.n_tokens(), 1);
        assert_eq!(batch.initialized_logits, vec![0]);

        Ok(())
    }

    #[test]
    fn drop_does_not_free_get_one_batch() {
        let tokens = vec![LlamaToken::new(1), LlamaToken::new(2)];
        let batch = LlamaBatch::get_one(&tokens).expect("test: get_one should succeed");

        assert_eq!(batch.allocated, 0);
        drop(batch);
    }

    #[test]
    fn checked_n_tokens_plus_one_as_usize_succeeds_for_zero() {
        let result = checked_n_tokens_plus_one_as_usize(0);

        assert_eq!(result, Ok(1));
    }

    #[test]
    fn checked_n_tokens_plus_one_as_usize_fails_for_negative() {
        let result = checked_n_tokens_plus_one_as_usize(-2);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn checked_n_tokens_plus_one_as_usize_fails_for_i32_max() {
        let result = checked_n_tokens_plus_one_as_usize(i32::MAX);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn checked_i32_as_usize_succeeds_for_zero() {
        let result = checked_i32_as_usize(0, "test_value");

        assert_eq!(result, Ok(0));
    }

    #[test]
    fn checked_i32_as_usize_fails_for_negative() {
        let result = checked_i32_as_usize(i32::MIN, "test_value");

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn checked_usize_as_llama_seq_id_succeeds_for_zero() {
        let result = checked_usize_as_llama_seq_id(0, "test_value");

        assert_eq!(result, Ok(0));
    }

    #[test]
    fn checked_usize_as_llama_seq_id_fails_for_overflow() {
        let result = checked_usize_as_llama_seq_id(usize::MAX, "test_value");

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn checked_usize_as_i32_succeeds_for_zero() {
        let result = checked_usize_as_i32(0, "test_value");

        assert_eq!(result, Ok(0));
    }

    #[test]
    fn checked_usize_as_i32_fails_for_overflow() {
        let result = checked_usize_as_i32(usize::MAX, "test_value");

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn checked_usize_as_llama_pos_succeeds_for_zero() {
        let result = checked_usize_as_llama_pos(0, "test_value");

        assert_eq!(result, Ok(0));
    }

    #[test]
    fn checked_usize_as_llama_pos_fails_for_overflow() {
        let result = checked_usize_as_llama_pos(usize::MAX, "test_value");

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn new_fails_for_oversized_n_tokens() {
        let result = LlamaBatch::new(usize::MAX, 1);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn add_fails_when_required_token_count_overflows_i32() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        batch.llama_batch.n_tokens = i32::MAX;

        let result = batch.add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn add_fails_when_existing_offset_is_negative() {
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        batch.llama_batch.n_tokens = -1;

        let result = batch.add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], false);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&BatchAddError::IntegerOverflow(String::new())),
        );
    }
}
