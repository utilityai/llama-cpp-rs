//! an rusty equivalent of `llama_token_data_array`.
use std::{ffi::CString, ptr};

use crate::{model::LlamaModel, token::data::LlamaTokenData};

use super::LlamaToken;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// the index of the selected token in ``data``
    pub selected: Option<usize>,
    /// is the data sorted?
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and whether or not the data is sorted.
    ///
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and whether or not the data is sorted.
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<T>(data: T, sorted: bool) -> LlamaTokenDataArray
    where
        T: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }
}

impl LlamaTokenDataArray {
    /// Modify the underlying data as a `llama_token_data_array`. and reconstruct the `LlamaTokenDataArray`.
    ///
    /// # Panics
    ///
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at runtime so breaking them is UB)
    ///
    /// SAFETY:
    /// [modify] cannot change the data pointer.
    /// if the data is not sorted, sorted must be false.
    /// the size of the data can only decrease (i.e you cannot add new elements).
    pub(crate) unsafe fn modify_as_c_llama_token_data_array<T>(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_sys_2::llama_token_data_array) -> T,
    ) -> T {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<llama_cpp_sys_2::llama_token_data>();

        let mut c_llama_token_data_array = llama_cpp_sys_2::llama_token_data_array {
            data,
            size,
            selected: self.selected.and_then(|s| s.try_into().ok()).unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);
        assert!(
            ptr::eq(data, c_llama_token_data_array.data),
            "data pointer changed"
        );
        assert!(c_llama_token_data_array.size <= size, "size increased");

        self.data.set_len(c_llama_token_data_array.size);
        self.sorted = c_llama_token_data_array.sorted;
        self.selected = c_llama_token_data_array
            .selected
            .try_into()
            .ok()
            .filter(|&s| s < self.data.len());

        result
    }

    pub(crate) unsafe fn apply_sampler(&mut self, sampler: *mut llama_cpp_sys_2::llama_sampler) {
        self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
            llama_cpp_sys_2::llama_sampler_apply(sampler, c_llama_token_data_array);
        });
    }

    pub(crate) unsafe fn apply_and_free_sampler(
        &mut self,
        sampler_fn: impl FnOnce() -> *mut llama_cpp_sys_2::llama_sampler,
    ) {
        let sampler = sampler_fn();
        self.apply_sampler(sampler);
        llama_cpp_sys_2::llama_sampler_free(sampler);
    }

    /// Modify the logits of [`Self`] in place using temperature sampling.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///     LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///     LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///     LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0)
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    ///
    /// candidates.sample_temp(0.5);
    ///
    /// assert_eq!(candidates.data[0].logit(), 0.2);
    /// assert_eq!(candidates.data[1].logit(), 0.4);
    /// assert_eq!(candidates.data[2].logit(), 1.4);
    /// ```
    pub fn sample_temp(&mut self, temperature: f32) {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_temp(temperature));
        }
    }

    /// Dynamic temperature implementation (a.k.a. entropy) described in the paper <https://arxiv.org/abs/2309.02772>.
    pub fn sample_temp_ext(&mut self, t: f32, delta: f32, exponent: f32) {
        unsafe {
            self.apply_and_free_sampler(|| {
                llama_cpp_sys_2::llama_sampler_init_temp_ext(t, delta, exponent)
            });
        }
    }

    /// Top-K sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    pub fn sample_top_k(&mut self, k: i32) {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_top_k(k));
        }
    }

    /// Locally Typical Sampling implementation described in the [paper](https://arxiv.org/abs/2202.00666).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///    LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///    LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///    LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_typical(0.5, 1);
    /// ```
    pub fn sample_typical(&mut self, p: f32, min_keep: usize) {
        unsafe {
            self.apply_and_free_sampler(|| {
                llama_cpp_sys_2::llama_sampler_init_typical(p, min_keep)
            });
        }
    }

    /// Nucleus sampling described in academic paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///   LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    ///
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_top_p(0.5, 1);
    ///
    /// assert_eq!(candidates.data.len(), 2);
    /// assert_eq!(candidates.data[0].id(), LlamaToken::new(2));
    /// assert_eq!(candidates.data[1].id(), LlamaToken::new(1));
    /// ```
    pub fn sample_top_p(&mut self, p: f32, min_keep: usize) {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_top_p(p, min_keep));
        }
    }

    /// Minimum P sampling as described in [#3841](https://github.com/ggerganov/llama.cpp/pull/3841)
    ///
    /// # Example
    ///
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    ///
    /// let candidates = vec![
    ///   LlamaTokenData::new(LlamaToken::new(4), -2., 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(1), 0.2, 0.0),
    ///   LlamaTokenData::new(LlamaToken::new(2), 0.7, 0.0),
    /// ];
    /// let mut candidates = LlamaTokenDataArray::from_iter(candidates, false);
    /// candidates.sample_min_p(0.1, 1);
    ///
    /// assert_eq!(candidates.data.len(), 3);
    /// ```
    pub fn sample_min_p(&mut self, p: f32, min_keep: usize) {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_min_p(p, min_keep));
        }
    }

    /// XTC sampling as described in <https://github.com/oobabooga/text-generation-webui/pull/6335>.
    pub fn sample_xtc(&mut self, p: f32, t: f32, min_keep: usize, seed: u32) {
        unsafe {
            self.apply_and_free_sampler(|| {
                llama_cpp_sys_2::llama_sampler_init_xtc(p, t, min_keep, seed)
            });
        }
    }

    /// This can be used to penalize certain patterns in the generated text, such as repeating the same token multiple times or using the same token too frequently.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_penalties(
        &mut self,
        tokens: &[LlamaToken],
        n_vocab: i32,
        special_eos_id: i32,
        linefeed_id: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
        penalize_nl: bool,
        ignore_eos: bool,
    ) {
        unsafe {
            self.apply_and_free_sampler(|| {
                let sampler = llama_cpp_sys_2::llama_sampler_init_penalties(
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

                for token in tokens {
                    llama_cpp_sys_2::llama_sampler_accept(sampler, token.0);
                }

                sampler
            });
        }
    }

    /// This can be used to penalize certain patterns in the generated text, such as repeating the same token multiple times or using the same token too frequently.
    pub fn sample_penalties_simple(
        &mut self,
        tokens: &[LlamaToken],
        model: &LlamaModel,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) {
        self.sample_penalties(
            tokens,
            model.n_vocab(),
            model.token_eos().0,
            model.token_nl().0,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            false,
            true,
        );
    }

    /// DRY sampler, designed by p-e-w, as described in: <https://github.com/oobabooga/text-generation-webui/pull/5677>, porting Koboldcpp implementation authored by pi6am: <https://github.com/LostRuins/koboldcpp/pull/982>
    #[allow(clippy::too_many_arguments)]
    pub fn sample_dry(
        &mut self,
        tokens: &[LlamaToken],
        model: &LlamaModel,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_penalty_last_n: i32,
        seq_breakers: &[impl AsRef<[u8]>],
    ) {
        let seq_breakers: Vec<CString> = seq_breakers
            .iter()
            .map(|s| {
                let bytes = s.as_ref();
                let null_byte = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
                CString::new(&bytes[..null_byte]).expect("Failed to slice away null bytes!")
            })
            .collect();

        let mut seq_breaker_pointers: Vec<*const i8> =
            seq_breakers.iter().map(|s| s.as_ptr()).collect();

        unsafe {
            self.apply_and_free_sampler(|| {
                let sampler = llama_cpp_sys_2::llama_sampler_init_dry(
                    model.model.as_ptr(),
                    dry_multiplier,
                    dry_base,
                    dry_allowed_length,
                    dry_penalty_last_n,
                    seq_breaker_pointers.as_mut_ptr(),
                    seq_breaker_pointers.len(),
                );

                for token in tokens {
                    llama_cpp_sys_2::llama_sampler_accept(sampler, token.0);
                }

                sampler
            });
        }
    }

    /// Randomly selects a token from the candidates based on their probabilities.
    pub fn sample_token(&mut self, seed: u32) -> LlamaToken {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_dist(seed));
        }
        self.selected_token()
            .expect("Dist sampler failed to select a token!")
    }

    /// Selects the token with the highest probability.
    pub fn sample_token_greedy(&mut self) -> LlamaToken {
        unsafe {
            self.apply_and_free_sampler(|| llama_cpp_sys_2::llama_sampler_init_greedy());
        }
        self.selected_token()
            .expect("Greedy sampler failed to select a token!")
    }
}
