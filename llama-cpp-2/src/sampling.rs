//! Safe wrapper around `llama_sampler`.
pub mod params;

use std::ffi::CString;
use std::fmt::{Debug, Formatter};

use crate::context::LlamaContext;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

use params::LlamaSamplerParams;

/// A safe wrapper around `llama_sampler`.
pub struct LlamaSampler {
    pub(crate) sampler: *mut llama_cpp_sys_2::llama_sampler,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChain").finish()
    }
}

unsafe fn new_inner(params: LlamaSamplerParams) -> *mut llama_cpp_sys_2::llama_sampler {
    match params {
        LlamaSamplerParams::Chain { no_perf, stages } => {
            let chain = llama_cpp_sys_2::llama_sampler_chain_init(
                llama_cpp_sys_2::llama_sampler_chain_params { no_perf },
            );

            for stage in stages {
                llama_cpp_sys_2::llama_sampler_chain_add(chain, new_inner(*stage));
            }

            chain
        }
        LlamaSamplerParams::Temp(p) => llama_cpp_sys_2::llama_sampler_init_temp(p),
        LlamaSamplerParams::TempExt { t, delta, exponent } => {
            llama_cpp_sys_2::llama_sampler_init_temp_ext(t, delta, exponent)
        }
        LlamaSamplerParams::TopK(k) => llama_cpp_sys_2::llama_sampler_init_top_k(k),
        LlamaSamplerParams::Typical { p, min_keep } => {
            llama_cpp_sys_2::llama_sampler_init_typical(p, min_keep)
        }
        LlamaSamplerParams::TopP { p, min_keep } => {
            llama_cpp_sys_2::llama_sampler_init_top_p(p, min_keep)
        }
        LlamaSamplerParams::MinP { p, min_keep } => {
            llama_cpp_sys_2::llama_sampler_init_min_p(p, min_keep)
        }
        LlamaSamplerParams::Xtc {
            p,
            t,
            min_keep,
            seed,
        } => llama_cpp_sys_2::llama_sampler_init_xtc(p, t, min_keep, seed),
        LlamaSamplerParams::Grammar {
            model,
            string,
            root,
        } => {
            let grammar_str = CString::new(string).unwrap();
            let grammar_root = CString::new(root).unwrap();
            llama_cpp_sys_2::llama_sampler_init_grammar(
                model.model.as_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
            )
        }
        LlamaSamplerParams::Dry {
            model,
            multiplier,
            base,
            allowed_length,
            penalty_last_n,
            seq_breakers,
        } => {
            let seq_breakers: Vec<CString> = seq_breakers
                .iter()
                .map(|s| CString::new(*s).unwrap())
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
        }
        LlamaSamplerParams::Penalties {
            n_vocab,
            special_eos_id,
            linefeed_id,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            penalize_nl,
            ignore_eos,
        } => llama_cpp_sys_2::llama_sampler_init_penalties(
            n_vocab,
            special_eos_id,
            linefeed_id,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            penalize_nl,
            ignore_eos,
        ),
        LlamaSamplerParams::Dist { seed } => llama_cpp_sys_2::llama_sampler_init_dist(seed),
        LlamaSamplerParams::Greedy => llama_cpp_sys_2::llama_sampler_init_greedy(),
    }
}

impl LlamaSampler {
    /// Create a new `LlamaSampler` from the given parameters.
    #[must_use]
    pub fn new(params: LlamaSamplerParams) -> Self {
        Self {
            sampler: unsafe { new_inner(params) },
        }
    }

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
    pub fn accept_many(&mut self, tokens: &[LlamaToken]) {
        for token in tokens {
            unsafe { llama_cpp_sys_2::llama_sampler_accept(self.sampler, token.0) }
        }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_free(self.sampler);
        }
    }
}
