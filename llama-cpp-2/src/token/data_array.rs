//! an rusty equivalent of `llama_token_data`.
use crate::context::LlamaContext;
use crate::token::data::LlamaTokenData;
use crate::token::LlamaToken;
use llama_cpp_sys_2::llama_token;
use std::cmp::min;
use std::ptr;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// is the data sorted?
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and weather or not the data is sorted.
    ///
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self { data, sorted }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and weather or not the data is sorted.
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<T>(data: T, sorted: bool) -> LlamaTokenDataArray
    where
        T: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }
}

impl LlamaTokenDataArray {
    /// Modify the underlying data as a `llama_token_data_array`. and reconstruct the `LlamaTokenDataArray`.
    ///
    /// # Panics
    ///
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at runtime so breaking them is UB)
    ///
    /// SAFETY:
    /// [modify] cannot change the data pointer.
    /// if the data is not sorted, sorted must be false.
    /// the size of the data can only decrease (i.e you cannot add new elements).
    pub(crate) unsafe fn modify_as_c_llama_token_data_array(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_sys_2::llama_token_data_array),
    ) {
        let size = self.data.len();
        let data = self.data.as_mut_ptr().cast();
        let mut c_llama_token_data_array = llama_cpp_sys_2::llama_token_data_array {
            data,
            size,
            sorted: self.sorted,
        };
        modify(&mut c_llama_token_data_array);
        assert!(
            ptr::eq(data, c_llama_token_data_array.data),
            "data pointer changed"
        );
        assert!(c_llama_token_data_array.size <= size, "size increased");
        self.data.set_len(c_llama_token_data_array.size);
        self.sorted = c_llama_token_data_array.sorted;
    }

    /// Repetition penalty described in [CTRL academic paper](https://arxiv.org/abs/1909.05858), with negative logit fix.
    /// Frequency and presence penalties described in [OpenAI API](https://platform.openai.com/docs/api-reference/parameter-details).
    ///
    /// # Parameters
    ///
    /// * `ctx` - the context to use. May be `None` if you do not care to record the sample timings.
    /// * `last_tokens` - the last tokens in the context.
    ///
    /// * `penalty_last_n` - the number of tokens back to consider for the repetition penalty. (0 for no penalty)
    /// * `penalty_repeat` - the repetition penalty. (1.0 for no penalty)
    /// * `penalty_freq` - the frequency penalty. (0.0 for no penalty)
    /// * `penalty_present` - the presence penalty. (0.0 for no penalty)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::collections::BTreeMap;
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let history = vec![
    ///   LlamaToken::new(2),
    ///   LlamaToken::new(1),
    ///   LlamaToken::new(0),
    /// ];
    ///
    /// let candidates = vec![
    ///    LlamaToken::new(0),
    ///    LlamaToken::new(1),
    ///    LlamaToken::new(2),
    ///    LlamaToken::new(3),
    /// ];
    ///
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates.iter().map(|&token| LlamaTokenData::new(token, 0.0, 0.0)), false);
    ///
    /// candidates.sample_repetition_penalty(None, &history, 2, 1.1, 0.1, 0.1);
    ///
    /// let token_logits = candidates.data.into_iter().map(|token_data| (token_data.id(), token_data.logit())).collect::<BTreeMap<_, _>>();
    /// assert_eq!(token_logits[&LlamaToken(0)], 0.0, "expected no penalty as it is out of `penalty_last_n`");
    /// assert!(token_logits[&LlamaToken(1)] < 0.0, "expected penalty as it is in `penalty_last_n`");
    /// assert!(token_logits[&LlamaToken(2)] < 0.0, "expected penalty as it is in `penalty_last_n`");
    /// assert_eq!(token_logits[&LlamaToken(3)], 0.0, "expected no penalty as it is not in `history`");
    /// ```
    pub fn sample_repetition_penalty(
        &mut self,
        ctx: Option<&mut LlamaContext>,
        last_tokens: &[LlamaToken],
        penalty_last_n: usize,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        let penalty_last_n = min(penalty_last_n, last_tokens.len().saturating_sub(1));
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_repetition_penalties(
                    ctx,
                    c_llama_token_data_array,
                    // safe cast as LlamaToken is repr(transparent)
                    last_tokens.as_ptr().cast::<llama_token>(),
                    penalty_last_n,
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                );
            });
        }
    }
}
