//! Sampling functions for the context.

use crate::context::LlamaContext;
use crate::grammar::LlamaGrammar;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

/// struct to hold params for sampling
#[derive(Debug)]
#[deprecated(
    since = "0.1.32",
    note = "this does not scale well with many params and does not allow for changing of orders."
)]
pub struct Sampler<'grammar> {
    token_data_array: LlamaTokenDataArray,
    grammar: Option<&'grammar mut LlamaGrammar>,
    temperature: Option<f32>,
}

impl<'grammar> Sampler<'grammar> {
    #[deprecated(
        since = "0.1.32",
        note = "this does not scale well with many params and does not allow for changing of orders."
    )]
    fn sample(self, llama_context: &mut LlamaContext) -> LlamaToken {
        match self {
            Sampler {
                token_data_array,
                grammar: None,
                temperature: None,
            } => llama_context.sample_token_greedy(token_data_array),
            Sampler {
                mut token_data_array,
                grammar: Some(grammar),
                temperature: None,
            } => {
                llama_context.sample_grammar(&mut token_data_array, grammar);
                let token = llama_context.sample_token_greedy(token_data_array);
                llama_context.grammar_accept_token(grammar, token);
                token
            }
            Sampler {
                mut token_data_array,
                grammar: None,
                temperature: Some(temp),
            } => {
                llama_context.sample_temp(&mut token_data_array, temp);
                llama_context.sample_token_softmax(&mut token_data_array);
                token_data_array.data[0].id()
            }
            Sampler {
                mut token_data_array,
                grammar: Some(grammar),
                temperature: Some(temperature),
            } => {
                llama_context.sample_grammar(&mut token_data_array, grammar);
                llama_context.sample_temp(&mut token_data_array, temperature);
                llama_context.sample_token_softmax(&mut token_data_array);
                let token = llama_context.sample_token_greedy(token_data_array);
                llama_context.grammar_accept_token(grammar, token);
                token
            }
        }
    }

    /// Create a new sampler.
    #[must_use]
    #[deprecated(
        since = "0.1.32",
        note = "this does not scale well with many params and does not allow for changing of orders."
    )]
    pub fn new(llama_token_data_array: LlamaTokenDataArray) -> Self {
        Self {
            token_data_array: llama_token_data_array,
            grammar: None,
            temperature: None,
        }
    }

    /// Set the grammar for sampling.
    #[must_use]
    #[deprecated(
        since = "0.1.32",
        note = "this does not scale well with many params and does not allow for changing of orders."
    )]
    pub fn with_grammar(mut self, grammar: &'grammar mut LlamaGrammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Set the temperature for sampling.
    ///
    /// ```
    /// # use llama_cpp_2::context::LlamaContext;
    /// # use llama_cpp_2::context::sample::Sampler;
    /// # use llama_cpp_2::grammar::LlamaGrammar;
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let _sampler = Sampler::new(LlamaTokenDataArray::new(vec![LlamaTokenData::new(LlamaToken(0), 0.0, 0.0)], false))
    ///     .with_temperature(0.5);
    /// ```
    #[must_use]
    #[deprecated(
        since = "0.1.32",
        note = "this does not scale well with many params and does not allow for changing of orders."
    )]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        if temperature == 0.0 {
            return self;
        }
        self.temperature = Some(temperature);
        self
    }
}

impl LlamaContext<'_> {
    /// Sample a token.
    ///
    /// # Panics
    ///
    /// - sampler contains no tokens
    #[deprecated(
        since = "0.1.32",
        note = "this does not scale well with many params and does not allow for changing of orders."
    )]
    pub fn sample(&mut self, sampler: Sampler) -> LlamaToken {
        sampler.sample(self)
    }

    /// Accept a token into the grammar.
    pub fn grammar_accept_token(&mut self, grammar: &mut LlamaGrammar, token: LlamaToken) {
        unsafe {
            llama_cpp_sys_2::llama_grammar_accept_token(
                self.context.as_ptr(),
                grammar.grammar.as_ptr(),
                token.0,
            );
        }
    }

    /// Perform grammar sampling.
    pub fn sample_grammar(
        &mut self,
        llama_token_data_array: &mut LlamaTokenDataArray,
        llama_grammar: &LlamaGrammar,
    ) {
        unsafe {
            llama_token_data_array.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_grammar(
                    self.context.as_ptr(),
                    c_llama_token_data_array,
                    llama_grammar.grammar.as_ptr(),
                );
            });
        }
    }

    /// Modify [`token_data`] in place using temperature sampling.
    ///
    /// # Panics
    ///
    /// - [`temperature`] is not between 0.0 and 1.0
    pub fn sample_temp(&self, token_data: &mut LlamaTokenDataArray, temperature: f32) {
        assert!(
            temperature >= 0.0,
            "temperature must be positive (was {temperature})"
        );
        assert!(
            temperature <= 1.0,
            "temperature must be less than or equal to 1.0 (was {temperature})"
        );
        if temperature == 0.0 {
            return;
        }
        let ctx: *mut llama_cpp_sys_2::llama_context = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_temp(ctx, c_llama_token_data_array, temperature);
            });
        }
    }

    /// Sample a token greedily.
    ///
    /// # Panics
    ///
    /// - [`token_data`] is empty
    #[must_use]
    pub fn sample_token_greedy(&self, mut token_data: LlamaTokenDataArray) -> LlamaToken {
        assert!(!token_data.data.is_empty(), "no tokens");
        let mut data_arr = llama_cpp_sys_2::llama_token_data_array {
            data: token_data
                .data
                .as_mut_ptr()
                .cast::<llama_cpp_sys_2::llama_token_data>(),
            size: token_data.data.len(),
            sorted: token_data.sorted,
        };
        let token = unsafe {
            llama_cpp_sys_2::llama_sample_token_greedy(
                self.context.as_ptr(),
                std::ptr::addr_of_mut!(data_arr),
            )
        };
        LlamaToken(token)
    }

    /// Tail Free Sampling described in [Tail-Free-Sampling](https://www.trentonbricken.com/Tail-Free-Sampling/).
    pub fn sample_tail_free(&self, token_data: &mut LlamaTokenDataArray, z: f32, min_keep: usize) {
        let ctx = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_tail_free(ctx, c_llama_token_data_array, z, min_keep);
            });
        }
    }

    /// Locally Typical Sampling implementation described in the [paper](https://arxiv.org/abs/2202.00666).
    pub fn sample_typical(&self, token_data: &mut LlamaTokenDataArray, p: f32, min_keep: usize) {
        let ctx = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_typical(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    /// Nucleus sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)"
    pub fn sample_top_p(&self, token_data: &mut LlamaTokenDataArray, p: f32, min_keep: usize) {
        let ctx = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_top_p(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    /// Minimum P sampling as described in [#3841](https://github.com/ggerganov/llama.cpp/pull/3841)
    pub fn sample_min_p(
        &self,
        llama_token_data: &mut LlamaTokenDataArray,
        p: f32,
        min_keep: usize,
    ) {
        let ctx = self.context.as_ptr();
        unsafe {
            llama_token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_min_p(ctx, c_llama_token_data_array, p, min_keep);
            });
        }
    }

    /// Top-K sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    pub fn sample_top_k(&self, token_data: &mut LlamaTokenDataArray, k: i32, min_keep: usize) {
        let ctx = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_top_k(ctx, c_llama_token_data_array, k, min_keep);
            });
        }
    }

    /// Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    pub fn sample_token_softmax(&self, token_data: &mut LlamaTokenDataArray) {
        let ctx = self.context.as_ptr();
        unsafe {
            token_data.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_sys_2::llama_sample_softmax(ctx, c_llama_token_data_array);
            });
        }
    }

    /// See [`LlamaTokenDataArray::sample_repetition_penalty`]
    pub fn sample_repetition_penalty(
        &mut self,
        token_data: &mut LlamaTokenDataArray,
        last_tokens: &[LlamaToken],
        penalty_last_n: usize,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) {
        token_data.sample_repetition_penalty(
            Some(self),
            last_tokens,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        );
    }
}
