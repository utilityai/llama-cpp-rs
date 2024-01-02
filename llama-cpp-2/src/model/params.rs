//! A safe wrapper around `llama_model_params`.

use std::fmt::Debug;

/// A safe wrapper around `llama_model_params`.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct LlamaModelParams {
    pub(crate) params: llama_cpp_sys_2::llama_model_params,
}

impl LlamaModelParams {
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// use mmap if possible
    #[must_use]
    pub fn use_mmap(&self) -> bool {
        self.params.use_mmap
    }

    /// force system to keep model in RAM
    #[must_use]
    pub fn use_mlock(&self) -> bool {
        self.params.use_mlock
    }

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        // The only way this conversion can fail is if u32 overflows the i32 - in which case we set
        // to MAX
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_2::model::params::LlamaModelParams;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), 0, "n_gpu_layers should be 0");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// ```
impl Default for LlamaModelParams {
    fn default() -> Self {
        LlamaModelParams {
            params: unsafe { llama_cpp_sys_2::llama_model_default_params() },
        }
    }
}
