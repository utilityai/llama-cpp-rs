//! Sampling functions for the context.

use crate::context::LlamaContext;
use crate::grammar::LlamaGrammar;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

#[cfg(feature = "sampler")]
pub mod sampler;

impl LlamaContext<'_> {
    /// Accept a token into the grammar.
    pub fn grammar_accept_token(&mut self, grammar: &mut LlamaGrammar, token: LlamaToken) {
        unsafe {
            llama_cpp_sys_2::llama_grammar_accept_token(
                grammar.grammar.as_ptr(),
                self.context.as_ptr(),
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

    /// See [`LlamaTokenDataArray::sample_temp`]
    pub fn sample_temp(&mut self, token_data: &mut LlamaTokenDataArray, temperature: f32) {
        token_data.sample_temp(Some(self), temperature);
    }

    /// Sample a token greedily. Note that this *does not* take into account anything that has modified the probabilities - it only looks at logits.
    ///
    /// Most of the time [`LlamaTokenDataArray::sample_softmax`] or [`LlamaTokenDataArray::sample_token`] should be used instead.
    ///
    /// # Panics
    ///
    /// - if `token_data` is empty
    #[must_use]
    pub fn sample_token_greedy(&mut self, mut token_data: LlamaTokenDataArray) -> LlamaToken {
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

    /// See [`LlamaTokenDataArray::sample_tail_free`]
    pub fn sample_tail_free(
        &mut self,
        token_data: &mut LlamaTokenDataArray,
        z: f32,
        min_keep: usize,
    ) {
        token_data.sample_tail_free(Some(self), z, min_keep);
    }

    /// See [`LlamaTokenDataArray::sample_typical`]
    pub fn sample_typical(
        &mut self,
        token_data: &mut LlamaTokenDataArray,
        p: f32,
        min_keep: usize,
    ) {
        token_data.sample_typical(Some(self), p, min_keep);
    }

    /// See [`LlamaTokenDataArray::sample_top_p`]
    pub fn sample_top_p(&mut self, token_data: &mut LlamaTokenDataArray, p: f32, min_keep: usize) {
        token_data.sample_top_p(Some(self), p, min_keep);
    }

    /// Minimum P sampling as described in [#3841](https://github.com/ggerganov/llama.cpp/pull/3841)
    pub fn sample_min_p(
        &mut self,
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

    /// See [`LlamaTokenDataArray::sample_top_k`]
    pub fn sample_top_k(&mut self, token_data: &mut LlamaTokenDataArray, k: i32, min_keep: usize) {
        token_data.sample_top_k(Some(self), k, min_keep);
    }

    /// See [`LlamaTokenDataArray::sample_softmax`]
    pub fn sample_token_softmax(&mut self, token_data: &mut LlamaTokenDataArray) {
        token_data.sample_softmax(Some(self));
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
