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
    pub(crate) unsafe fn modify_as_c_llama_token_data_array<T>(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_sys_2::llama_token_data_array) -> T,
    ) -> T {
        let size = self.data.len();
        let data = self.data.as_mut_ptr().cast();
        let mut c_llama_token_data_array = llama_cpp_sys_2::llama_token_data_array {
            data,
            size,
            sorted: self.sorted,
        };
        let result = modify(&mut c_llama_token_data_array);
        assert!(
            ptr::eq(data, c_llama_token_data_array.data),
            "data pointer changed"
        );
        assert!(c_llama_token_data_array.size <= size, "size increased");
        self.data.set_len(c_llama_token_data_array.size);
        self.sorted = c_llama_token_data_array.sorted;
        result
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

    /// Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let lowest = LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0);
    /// let middle = LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0);
    /// let highest = LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0);
    ///
    /// let candidates = vec![lowest, middle, highest];
    ///
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_softmax(None);
    ///
    /// assert!(candidates.sorted);
    /// assert_eq!(candidates.data[0].id(), highest.id());
    /// assert_eq!(candidates.data[0].logit(), highest.logit());
    /// assert!(candidates.data[0].p() > candidates.data[1].p());
    /// assert_eq!(candidates.data[1].id(), middle.id());
    /// assert_eq!(candidates.data[1].logit(), middle.logit());
    /// assert!(candidates.data[1].p() > candidates.data[2].p());
    /// assert_eq!(candidates.data[2].id(), lowest.id());
    /// assert_eq!(candidates.data[2].logit(), lowest.logit());
    /// ```
    pub fn sample_softmax(&mut self, ctx: Option<&mut LlamaContext>) {
        unsafe {
            let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_softmax(ctx, c_llama_token_data_array);
            });
        }
    }

    /// Modify the logits of [`Self`] in place using temperature sampling.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///     LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///     LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///     LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0)
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    ///
    /// candidates.sample_temp(None, 0.5);
    ///
    /// assert_ne!(candidates.data[0].logit(), 0.1);
    /// assert_ne!(candidates.data[1].logit(), 0.2);
    /// assert_ne!(candidates.data[2].logit(), 0.7);
    /// ```
    pub fn sample_temp(&mut self, ctx: Option<&mut LlamaContext>, temperature: f32) {
        if temperature == 0.0 {
            return;
        }
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_temp(ctx, c_llama_token_data_array, temperature);
            });
        }
    }

    /// Randomly selects a token from the candidates based on their probabilities.
    pub fn sample_token(&mut self, ctx: &mut LlamaContext) -> LlamaToken {
        let llama_token = unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_token(ctx.context.as_ptr(), c_llama_token_data_array)
            })
        };
        LlamaToken(llama_token)
    }

    /// Top-K sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    pub fn sample_top_k(&mut self, ctx: Option<&mut LlamaContext>, k: i32, min_keep: usize) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_top_k(ctx, c_llama_token_data_array, k, min_keep);
            });
        }
    }

    /// Tail Free Sampling described in [Tail-Free-Sampling](https://www.trentonbricken.com/Tail-Free-Sampling/).
    pub fn sample_tail_free(&mut self, ctx: Option<&mut LlamaContext>, z: f32, min_keep: usize) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_tail_free(ctx, c_llama_token_data_array, z, min_keep);
            });
        }
    }

    /// Locally Typical Sampling implementation described in the [paper](https://arxiv.org/abs/2202.00666).
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///    LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///    LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///    LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_typical(None, 0.5, 1);
    ///
    /// ```
    pub fn sample_typical(&mut self, ctx: Option<&mut LlamaContext>, p: f32, min_keep: usize) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_typical(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    /// Nucleus sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///   LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    ///
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_top_p(None, 0.5, 1);
    ///
    /// assert_eq!(candidates.data.len(), 2);
    /// assert_eq!(candidates.data[0].id(), LlamaToken::new(2));
    /// assert_eq!(candidates.data[1].id(), LlamaToken::new(1));
    /// ```
    pub fn sample_top_p(&mut self, ctx: Option<&mut LlamaContext>, p: f32, min_keep: usize) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_top_p(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    /// Minimum P sampling as described in [#3841](https://github.com/ggerganov/llama.cpp/pull/3841)
    ///
    /// # Example
    ///
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///   LlamaTokenData::new(LlamaToken::new(4), 0.0001, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_min_p(None, 0.05, 1);
    /// ```
    pub fn sample_min_p(&mut self, ctx: Option<&mut LlamaContext>, p: f32, min_keep: usize) {
        let ctx = ctx.map_or(ptr::null_mut(), |ctx| ctx.context.as_ptr());
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_min_p(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    ///  Mirostat 2.0 algorithm described in the [paper](https://arxiv.org/abs/2007.14966). Uses tokens instead of words.
    ///
    /// # Parameters
    ///
    /// * `tau`  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// * `eta` The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// * `mu` Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    pub fn sample_token_mirostat_v2(
        &mut self,
        ctx: &mut LlamaContext,
        tau: f32,
        eta: f32,
        mu: &mut f32,
    ) -> LlamaToken {
        let mu_ptr = ptr::from_mut(mu);
        let token = unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_token_mirostat_v2(
                    ctx.context.as_ptr(),
                    c_llama_token_data_array,
                    tau,
                    eta,
                    mu_ptr,
                )
            })
        };
        *mu = unsafe { *mu_ptr };
        LlamaToken(token)
    }

    ///  Mirostat 1.0 algorithm described in the [paper](https://arxiv.org/abs/2007.14966). Uses tokens instead of words.
    ///
    /// # Parameters
    ///
    /// * `tau`  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// * `eta`  The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// * `m`  The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// * `mu`  Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    pub fn sample_token_mirostat_v1(
        &mut self,
        ctx: &mut LlamaContext,
        tau: f32,
        eta: f32,
        m: i32,
        mu: &mut f32,
    ) -> LlamaToken {
        let mu_ptr = ptr::from_mut(mu);
        let token = unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_token_mirostat(
                    ctx.context.as_ptr(),
                    c_llama_token_data_array,
                    tau,
                    eta,
                    m,
                    mu_ptr,
                )
            })
        };
        *mu = unsafe { *mu_ptr };
        LlamaToken(token)
    }
}
