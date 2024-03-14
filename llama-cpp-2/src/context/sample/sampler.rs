//! A more rusty way of sampling. Allows for adding a stack of sampling steps and a `finalizer` which selects a token from the remaining candidates.
//!
//! # Example
//!
//! ```rust
//! use llama_cpp_2::context::sample::sampler::Sampler;
//! use llama_cpp_2::token::data::LlamaTokenData;
//! use llama_cpp_2::token::data_array::LlamaTokenDataArray;
//! use llama_cpp_2::token::LlamaToken;
//!
//! let mut history = vec![];
//! let candidates = LlamaTokenDataArray::from_iter((0..4).map(|i| LlamaTokenData::new(LlamaToken::new(i), i as f32 / 6.0, 0.0)), false);
//!
//! let token = {
//!   let mut sampler = Sampler::greedy();
//!   sampler.push_sample_repetition_penalty_step(&history, 64, 1.1, 0.0, 0.0);
//!   sampler.push_top_k_step(40, 1);
//!   sampler.push_sample_tail_free_step(1.0, 1);
//!   sampler.push_sample_typical_step(1.0, 1);
//!   sampler.push_sample_top_p_step(0.95, 1);
//!   sampler.push_min_p_step(0.05, 1);
//!   sampler.push_temperature_step(0.5);
//!   sampler.sample(candidates)
//! };
//! history.push(token[0].id());
//!
//! println!("{:?}", token);
//! ```

use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use std::fmt::{Debug, Formatter};

/// A single step to sample tokens from the remaining candidates.
pub type SampleStep<'a> = Box<dyn FnMut(&mut LlamaTokenDataArray) + 'a>;

/// The final step to select one or more tokens from the remaining candidates.
pub type SampleFinalizer<'a> = Box<dyn FnMut(LlamaTokenDataArray) -> Vec<LlamaTokenData> + 'a>;

/// A series of sampling steps that will produce a vector of token data.
///
/// [`a`] is the lifetime of captured references in the steps and finalizer.
#[non_exhaustive]
pub struct Sampler<'a> {
    /// The steps to take when sampling.
    pub steps: Vec<SampleStep<'a>>,
    /// The final step to select one or more tokens from the remaining candidates.
    pub finalizer: SampleFinalizer<'a>,
}

impl Debug for Sampler<'_> {
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

impl<'a> Sampler<'a> {
    /// Create a very simple sampler that selects a single token with the greatest logit (greedy sampling).
    ///
    /// # Example
    ///
    /// ```rust
    /// use llama_cpp_2::context::sample::sampler::Sampler;
    /// use llama_cpp_2::token::data::LlamaTokenData;
    /// use llama_cpp_2::token::data_array::LlamaTokenDataArray;use llama_cpp_2::token::LlamaToken;
    ///
    /// let mut sampler = Sampler::greedy();
    ///
    /// let candidates = (0..4).map(|i| LlamaTokenData::new(LlamaToken::new(i), i as f32 / 6.0, 0.0));
    /// let tokens = sampler.sample(LlamaTokenDataArray::from_iter(candidates, false));
    /// assert_eq!(tokens[0].id(), LlamaToken::new(3));
    /// ```
    #[must_use]
    pub fn greedy() -> Self {
        let finalizer= |mut token_data: LlamaTokenDataArray| {
            if token_data.data.is_empty() {
                return vec![];
            }
            if token_data.sorted {
                vec![token_data.data[0]]
            } else {
                token_data.sample_softmax(None);
                vec![token_data.data[0]]
            }
        };
        Self::new(finalizer)
    }

    /// Adds a repetition penalty sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_repetition_penalty`]
    pub fn push_sample_repetition_penalty_step(
        &mut self,
        history: &'a [LlamaToken],
        penalty_last_n: usize,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_repetition_penalty(
                    None,
                    history,
                    penalty_last_n,
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                );
            }));
    }

    /// Adds a typical sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_typical`]
    pub fn push_sample_typical_step(&mut self, p: f32, min_keep: usize) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_typical(None, p, min_keep);
            }));
    }

    /// Adds a Top-p sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_top_p`]
    pub fn push_sample_top_p_step(&mut self, p: f32, min_keep: usize) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_top_p(None, p, min_keep);
            }));
    }

    /// Adds a tail-free sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_tail_free`]
    pub fn push_sample_tail_free_step(&mut self, z: f32, min_keep: usize) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_tail_free(None, z, min_keep);
            }));
    }

    /// Adds a top-k sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_top_k`]
    pub fn push_top_k_step(&mut self, k: i32, min_keep: usize) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_top_k(None, k, min_keep);
            }));
    }

    /// Adds a temperature sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_temp`]
    pub fn push_temperature_step(&mut self, temperature: f32) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_temp(None, temperature);
            }));
    }

    /// Adds a minimum P sampling step to the sampler.
    ///
    /// See [`LlamaTokenDataArray::sample_min_p`]
    pub fn push_min_p_step(&mut self, p: f32, min_keep: usize) {
        self.steps
            .push(Box::new(move |can: &mut LlamaTokenDataArray| {
                can.sample_min_p(None, p, min_keep);
            }));
    }

    /// Create a new sampler with a given finalizer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llama_cpp_2::context::sample::sampler::Sampler;
    /// use llama_cpp_2::token::data::LlamaTokenData;
    /// use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// use llama_cpp_2::token::LlamaToken;
    ///
    /// // a very silly way to sample.
    /// let always_0 = |can: LlamaTokenDataArray| -> Vec<LlamaTokenData> { can.data.into_iter().filter(|t| t.id() == LlamaToken::new(0)).collect::<Vec<_>>() };
    ///
    /// let mut sampler = Sampler::new(always_0);
    ///
    /// let candidates = (0..4).map(|i| LlamaTokenData::new(LlamaToken::new(i), i as f32, 0.0));
    ///
    /// let token = sampler.sample(LlamaTokenDataArray::from_iter(candidates, false));
    /// assert_eq!(token[0].id(), LlamaToken::new(0));
    ///
    /// ```
    pub fn new(
        finalizer: impl FnMut(LlamaTokenDataArray) -> Vec<LlamaTokenData> + 'a,
    ) -> Self {
        Self {
            steps: Vec::new(),
            finalizer: Box::new(finalizer),
        }
    }

    /// Adds a step to the sampler.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llama_cpp_2::context::sample::sampler::Sampler;
    /// use llama_cpp_2::token::data::LlamaTokenData;
    /// use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// use llama_cpp_2::token::LlamaToken;
    ///
    /// let mut favor_even_tokens = |can: &mut LlamaTokenDataArray| {
    ///    for token in can.data.iter_mut() {
    ///        if token.id().0 % 2 == 0 {
    ///           token.set_logit(token.logit() + 1.0);
    ///       }
    ///    }
    /// };
    /// let mut sampler = Sampler::greedy();
    /// sampler.push_step(favor_even_tokens);
    ///
    /// let candidates = (0..4).map(|i| LlamaTokenData::new(LlamaToken::new(i), i as f32, 0.0));
    ///
    /// let token = sampler.sample(LlamaTokenDataArray::from_iter(candidates, false));
    ///
    /// assert_eq!(token[0].id(), LlamaToken::new(2));
    /// ```
    pub fn push_step(&mut self, step: impl FnMut(&mut LlamaTokenDataArray) + 'a) {
        self.steps.push(Box::new(step));
    }

    /// Sample a token from the given candidates.
    #[must_use]
    pub fn sample(&mut self, mut candidates: LlamaTokenDataArray) -> Vec<LlamaTokenData> {
        for step in &mut self.steps {
            step(&mut candidates);
        }
        (self.finalizer)(candidates)
    }
}
