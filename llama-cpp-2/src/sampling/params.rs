//! Safe wrapper around `llama_sampler_chain_params`.

use std::fmt::{Debug, Formatter};

/// A safe wrapper around `llama_sampler`.
pub struct LlamaSamplerChainParams {
    pub(crate) sampler_chain_params: llama_cpp_sys_2::llama_sampler_chain_params,
}

impl Debug for LlamaSamplerChainParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChainParams").finish()
    }
}

impl Default for LlamaSamplerChainParams {
    fn default() -> Self {
        let sampler_chain_params = unsafe { llama_cpp_sys_2::llama_sampler_chain_default_params() };

        Self {
            sampler_chain_params,
        }
    }
}

impl LlamaSamplerChainParams {
    /// Set whether to measure performance timings
    #[must_use]
    pub fn with_no_perf(mut self, no_perf: bool) -> Self {
        self.sampler_chain_params.no_perf = no_perf;
        self
    }

    /// Get whether to measure performance timings
    #[must_use]
    pub fn no_perf(&self) -> bool {
        self.sampler_chain_params.no_perf
    }
}
