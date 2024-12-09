//! Safe wrapper around `llama_sampler`.

use std::ffi::CString;
use std::fmt::{Debug, Formatter};

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

/// A safe wrapper around `llama_sampler`.
pub struct LlamaSampler {
    pub(crate) sampler: *mut llama_cpp_sys_2::llama_sampler,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChain").finish()
    }
}

impl LlamaSampler {
    /// Sample and accept a token from the idx-th output of the last evaluation
    #[must_use]
    pub fn sample(&self, ctx: &LlamaContext, idx: i32) -> LlamaToken {
        let token = unsafe {
            llama_cpp_sys_2::llama_sampler_sample(self.sampler, ctx.context.as_ptr(), idx)
        };

        LlamaToken(token)
    }

    /// Applies this sampler to a [`LlamaTokenDataArray`].
    pub fn apply(&mut self, data_array: &mut LlamaTokenDataArray) {
        data_array.apply_sampler(self);
    }

    /// Accepts a token from the sampler, possibly updating the internal state of certain samplers
    /// (e.g. grammar, repetition, etc.)
    pub fn accept(&mut self, token: LlamaToken) {
        unsafe { llama_cpp_sys_2::llama_sampler_accept(self.sampler, token.0) }
    }

    /// Accepts several tokens from the sampler or context, possibly updating the internal state of
    /// certain samplers (e.g. grammar, repetition, etc.)
    pub fn accept_many(&mut self, tokens: impl IntoIterator<Item = impl AsRef<LlamaToken>>) {
        for token in tokens {
            unsafe { llama_cpp_sys_2::llama_sampler_accept(self.sampler, token.as_ref().0) }
        }
    }

    /// Accepts several tokens from the sampler or context, possibly updating the internal state of
    /// certain samplers (e.g. grammar, repetition, etc.)
    #[must_use]
    pub fn with_tokens(mut self, tokens: impl IntoIterator<Item = impl AsRef<LlamaToken>>) -> Self {
        self.accept_many(tokens);
        self
    }

    #[must_use]
    pub fn chain(samplers: impl IntoIterator<Item = Self>, no_perf: bool) -> Self {
        unsafe {
            let chain = llama_cpp_sys_2::llama_sampler_chain_init(
                llama_cpp_sys_2::llama_sampler_chain_params { no_perf },
            );

            for sampler in samplers {
                llama_cpp_sys_2::llama_sampler_chain_add(chain, sampler.sampler);

                // Do not call `llama_sampler_free` on the sampler, as the internal sampler is now
                // owned by the chain
                std::mem::forget(sampler);
            }

            Self { sampler: chain }
        }
    }

    /// Same as [`Self::chain`] with `no_perf = false`.
    #[must_use]
    pub fn chain_simple(samplers: impl IntoIterator<Item = Self>) -> Self {
        Self::chain(samplers, false)
    }

    #[must_use]
    pub fn temp(t: f32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_temp(t) };
        Self { sampler }
    }

    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_temp_ext(t, delta, exponent) };
        Self { sampler }
    }

    #[must_use]
    pub fn top_k(k: i32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_top_k(k) };
        Self { sampler }
    }

    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_typical(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_top_p(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_min_p(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_xtc(p, t, min_keep, seed) };
        Self { sampler }
    }

    #[must_use]
    pub fn grammar(model: &LlamaModel, grammar_str: &str, grammar_root: &str) -> Self {
        let grammar_str = CString::new(grammar_str).unwrap();
        let grammar_root = CString::new(grammar_root).unwrap();

        let sampler = unsafe {
            llama_cpp_sys_2::llama_sampler_init_grammar(
                model.model.as_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
            )
        };
        Self { sampler }
    }

    #[must_use]
    pub fn dry(
        model: &LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: &[impl AsRef<[u8]>],
    ) -> Self {
        let sampler = unsafe {
            let seq_breakers: Vec<CString> = seq_breakers
                .iter()
                .map(|s| CString::new(s.as_ref()).unwrap())
                .collect();
            let mut seq_breaker_pointers: Vec<*const i8> =
                seq_breakers.iter().map(|s| s.as_ptr()).collect();
            llama_cpp_sys_2::llama_sampler_init_dry(
                model.model.as_ptr(),
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                seq_breaker_pointers.as_mut_ptr(),
                seq_breaker_pointers.len(),
            )
        };
        Self { sampler }
    }

    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn penalties(
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
        let sampler = unsafe {
            llama_cpp_sys_2::llama_sampler_init_penalties(
                n_vocab,
                special_eos_id,
                linefeed_id,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
                penalize_nl,
                ignore_eos,
            )
        };
        Self { sampler }
    }

    /// Same as [`Self::penalties`], but with `n_vocab`, `special_eos_id`, and `linefeed_id`
    /// initialized from `model`, `penalize_nl = false`, and `ignore_eos = true`.
    #[must_use]
    pub fn penalties_simple(
        model: &LlamaModel,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        Self::penalties(
            model.n_vocab(),
            model.token_eos().0,
            model.token_nl().0,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            false,
            true,
        )
    }

    #[must_use]
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        let sampler =
            unsafe { llama_cpp_sys_2::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) };
        Self { sampler }
    }

    #[must_use]
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_mirostat_v2(seed, tau, eta) };
        Self { sampler }
    }

    #[must_use]
    pub fn dist(seed: u32) -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_dist(seed) };
        Self { sampler }
    }

    #[must_use]
    pub fn greedy() -> Self {
        let sampler = unsafe { llama_cpp_sys_2::llama_sampler_init_greedy() };
        Self { sampler }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_free(self.sampler);
        }
    }
}
