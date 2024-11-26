//! Safe wrapper around `llama_sampler`.
pub mod params;

use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;
use crate::LlamaSamplerError;

/// A safe wrapper around `llama_sampler`.
pub struct LlamaSampler {
    pub(crate) sampler: NonNull<llama_cpp_sys_2::llama_sampler>,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChain").finish()
    }
}

impl LlamaSampler {
    /// Create a new `LlamaSampler` from the given parameters.
    /// # Errors
    /// Returns an error if the underlying C++ code returns a null pointer.
    pub fn new(params: params::LlamaSamplerChainParams) -> Result<Self, LlamaSamplerError> {
        let sampler = unsafe {
            NonNull::new(llama_cpp_sys_2::llama_sampler_chain_init(
                params.sampler_chain_params,
            ))
            .ok_or(LlamaSamplerError::NullReturn)
        }?;

        Ok(Self { sampler })
    }

    /// Samples the token with the largest probability.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_greedy(mut self) -> Self {
        unsafe {
            let greedy_sampler = llama_cpp_sys_2::llama_sampler_init_greedy();
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), greedy_sampler);
        }

        self
    }

    /// Samples according to the probability distribution of the tokens.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_dist(mut self, seed: u32) -> Self {
        unsafe {
            let dist_sampler = llama_cpp_sys_2::llama_sampler_init_dist(seed);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), dist_sampler);
        }

        self
    }

    /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_top_k(mut self, k: i32) -> Self {
        unsafe {
            let top_k_sampler = llama_cpp_sys_2::llama_sampler_init_top_k(k);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), top_k_sampler);
        }

        self
    }

    /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" <https://arxiv.org/abs/1904.09751>
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_top_p(mut self, p: f32, min_keep: usize) -> Self {
        unsafe {
            let top_p_sampler = llama_cpp_sys_2::llama_sampler_init_top_p(p, min_keep);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), top_p_sampler);
        }

        self
    }

    /// Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_min_p(mut self, p: f32, min_keep: usize) -> Self {
        unsafe {
            let min_p_sampler = llama_cpp_sys_2::llama_sampler_init_min_p(p, min_keep);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), min_p_sampler);
        }

        self
    }

    /// Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_typical(mut self, p: f32, min_keep: usize) -> Self {
        unsafe {
            let typical_sampler = llama_cpp_sys_2::llama_sampler_init_typical(p, min_keep);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), typical_sampler);
        }

        self
    }

    /// Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_temp(mut self, t: f32) -> Self {
        unsafe {
            let temp_sampler = llama_cpp_sys_2::llama_sampler_init_temp(t);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_sampler);
        }

        self
    }

    /// Dynamic temperature implementation (a.k.a. entropy) described in the paper <https://arxiv.org/abs/2309.02772>.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_temp_ext(mut self, t: f32, delta: f32, exponent: f32) -> Self {
        unsafe {
            let temp_ext_sampler = llama_cpp_sys_2::llama_sampler_init_temp_ext(t, delta, exponent);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_ext_sampler);
        }

        self
    }

    /// Mirostat 1.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Arguments
    ///
    /// * `candidates` - A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// * `tau` -  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// * `eta` - The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// * `m` - The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// * `mu` - Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_mirostat(mut self, n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        unsafe {
            let temp_ext_sampler =
                llama_cpp_sys_2::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_ext_sampler);
        }

        self
    }

    /// Mirostat 2.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Arguments
    ///
    /// * `candidates` - A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// * `tau` -  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// * `eta` - The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// * `mu` - Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_mirostat_v2(mut self, seed: u32, tau: f32, eta: f32) -> Self {
        unsafe {
            let temp_ext_sampler = llama_cpp_sys_2::llama_sampler_init_mirostat_v2(seed, tau, eta);
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_ext_sampler);
        }

        self
    }

    /// Samples constrained by a context-free grammar in the GGML BNF (GBNF) format.
    ///
    /// # Panics
    /// Panics if a provided string contains a null byte.
    #[must_use]
    #[allow(unused_mut)]
    pub fn add_grammar(
        mut self,
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Self {
        unsafe {
            let grammar_str = CString::new(grammar_str).unwrap();
            let grammar_root = CString::new(grammar_root).unwrap();
            let grammar_sampler = llama_cpp_sys_2::llama_sampler_init_grammar(
                model.model.as_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
            );
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), grammar_sampler);
        }

        self
    }

    /// Adds penalties to the sampler. This can be used to penalize certain patterns in the generated text, such as repeating the same token multiple times or using the same token too frequently.
    #[allow(unused_mut, clippy::too_many_arguments)]
    #[must_use]
    pub fn add_penalties(
        mut self,
        n_vocab: i32,
        special_eos_id: i32,
        linefeed_id: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
        penalize_nl: bool,
        ignore_eos: bool,
    ) -> Self {
        unsafe {
            let temp_ext_sampler = llama_cpp_sys_2::llama_sampler_init_penalties(
                n_vocab,
                special_eos_id,
                linefeed_id,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
                penalize_nl,
                ignore_eos,
            );
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_ext_sampler);
        }

        self
    }

    /// Sample and accept a token from the idx-th output of the last evaluation
    #[must_use]
    pub fn sample(&self, ctx: &LlamaContext, idx: i32) -> LlamaToken {
        let token = unsafe {
            llama_cpp_sys_2::llama_sampler_sample(self.sampler.as_ptr(), ctx.context.as_ptr(), idx)
        };

        LlamaToken(token)
    }

    /// Accepts a token from the sampler, possibly updating the internal state of certain samplers (e.g. grammar, repetition, etc.)
    pub fn accept(&mut self, token: LlamaToken) {
        unsafe { llama_cpp_sys_2::llama_sampler_accept(self.sampler.as_ptr(), token.0) }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_free(self.sampler.as_ptr());
        }
    }
}
