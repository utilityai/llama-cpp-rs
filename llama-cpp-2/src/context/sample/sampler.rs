//! Create a sampler struct to encapsulate the sampling process. This allows passing all the possible
//! sampling parameters around as a single struct, and also allow late binding of expensive context
//! like [`crate::context::LlamaContext`] or token history to the sampler.
//!
//! # Example
//! 
//! **Llama.cpp default sampler**
//!
//! ```rust
//! use llama_cpp_2::context::sample::sampler::{Sampler, SampleStep};
//! use llama_cpp_2::token::data::LlamaTokenData;
//! use llama_cpp_2::token::data_array::LlamaTokenDataArray;
//! use llama_cpp_2::token::LlamaToken;
//!
//! // Sample a token greedily and add to the history.
//! let mut finalizer = &|mut canidates: LlamaTokenDataArray, history: &mut Vec<LlamaToken>| {
//!     canidates.sample_softmax(None);
//!     let token = canidates.data[0];
//!     history.push(token.id());
//!     vec![token]
//! };
//!
//! let mut history = vec![];
//! let mut sampler = Sampler::new(finalizer);
//!
//! sampler.push_step(&|c, history| c.sample_repetition_penalty(None, history, 64, 1.1, 0.0, 0.0));
//! sampler.push_step(&|c, _| c.sample_top_k(None, 40, 1));
//! sampler.push_step(&|c, _| c.sample_tail_free(None, 1.0, 1));
//! sampler.push_step(&|c, _| c.sample_typical(None, 1.0, 1));
//! sampler.push_step(&|c, _| c.sample_top_p(None, 0.95, 1));
//! sampler.push_step(&|c, _| c.sample_min_p(None, 0.05, 1));
//! sampler.push_step(&|c, _| c.sample_temp(None, 0.5));
//!
//! // random candidates
//! let candidates = LlamaTokenDataArray::from_iter((0..4).map(|i| LlamaTokenData::new(LlamaToken::new(i), i as f32 / 6.0, 0.0)), false);
//!
//! for _ in 0..10 {
//!    let tokens = sampler.sample(&mut history, candidates.clone());
//!    assert_eq!(tokens.len(), 1);
//! }
//!
//! assert_eq!(history.len(), 10);
//! ```

use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use std::fmt::{Debug, Formatter};

/// A single step to sample tokens from the remaining candidates.
pub type SampleStep<C> = dyn Fn(&mut LlamaTokenDataArray, &mut C);

/// The final step to select tokens from the remaining candidates.
pub type SampleFinalizer<C> = dyn Fn(LlamaTokenDataArray, &mut C) -> Vec<LlamaTokenData>;

/// A series of sampling steps that will produce a vector of token data.
///
/// `C` is dynamic context that will be passed to the sampling functions. Some sampling steps may
/// require state to be maintained across multiple samples, and this context can be used to store
/// that state. For example, [`LlamaTokenDataArray::sample_token_mirostat_v2`] requires a `mu` to be
/// shared across multiple samples.
pub struct Sampler<'a, C> {
    /// The steps to take when sampling.
    pub steps: Vec<&'a SampleStep<C>>,
    /// The final step to select one or more tokens from the remaining candidates.
    pub finalizer: &'a SampleFinalizer<C>,
}

impl<T> Debug for Sampler<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field(
                "steps",
                &format!(
                    "{} steps of Box<dyn FnMut(&mut LlamaTokenDataArray) -> ()>",
                    &self.steps.len()
                ),
            )
            .field(
                "finalizer",
                &"Box<dyn FnMut(LlamaTokenDataArray) -> Vec<LlamaTokenData>>",
            )
            .finish()
    }
}

impl<'a, T> Sampler<'a, T> {
    /// Create a new sampler with a given finalizer.
    pub fn new(finalizer: &'a SampleFinalizer<T>) -> Self {
        Self {
            steps: vec![],
            finalizer,
        }
    }

    /// Adds a step to the sampler.
    pub fn push_step(&mut self, step: &'a SampleStep<T>) {
        self.steps.push(step);
    }

    /// Sample a token from the given candidates.
    #[must_use]
    pub fn sample(
        &mut self,
        context: &mut T,
        mut candidates: LlamaTokenDataArray,
    ) -> Vec<LlamaTokenData> {
        for step in &self.steps {
            step(&mut candidates, context);
        }
        (self.finalizer)(candidates, context)
    }
}
