//! Safe wrapper around `llama_sampler`.
pub mod params;

use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

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
    pub fn new(params: params::LlamaSamplerChainParams) -> Result<Self, LlamaSamplerError> {
        let sampler = unsafe {
            NonNull::new(llama_cpp_sys_2::llama_sampler_chain_init(
                params.sampler_chain_params,
            ))
            .ok_or(LlamaSamplerError::NullReturn)
        }?;

        Ok(Self { sampler })
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_free(self.sampler.as_ptr());
        }
    }
}
