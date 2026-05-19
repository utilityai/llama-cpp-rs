//! an rusty equivalent of `llama_token_data_array`.
use std::ptr;

use crate::error::TokenSamplingError;
use crate::sampling::LlamaSampler;
use crate::token::data::LlamaTokenData;

use super::LlamaToken;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
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
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// # use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub const fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and whether or not the data is sorted.
    /// ```
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// # use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<TIterator>(data: TIterator, sorted: bool) -> Self
    where
        TIterator: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    /// Returns the current selected token, if one exists.
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
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at
    /// runtime so breaking them is UB)
    ///
    /// # Safety
    ///
    /// The returned array formed by the data pointer and the length must entirely consist of
    /// initialized token data and the length must be less than the capacity of this array's data
    /// buffer.
    /// If the data is not sorted, sorted must be false.
    pub unsafe fn modify_as_c_llama_token_data_array<TResult>(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_bindings_sys::llama_token_data_array) -> TResult,
    ) -> TResult {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token_data>();

        let mut c_llama_token_data_array = llama_cpp_bindings_sys::llama_token_data_array {
            data,
            size,
            selected: self
                .selected
                .and_then(|selected_index| selected_index.try_into().ok())
                .unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);

        assert!(c_llama_token_data_array.size <= self.data.capacity());
        // SAFETY: caller guarantees the returned data and size are valid.
        unsafe {
            if !ptr::eq(c_llama_token_data_array.data, data) {
                ptr::copy(
                    c_llama_token_data_array.data,
                    data,
                    c_llama_token_data_array.size,
                );
            }
            self.data.set_len(c_llama_token_data_array.size);
        }

        self.sorted = c_llama_token_data_array.sorted;
        self.selected = c_llama_token_data_array
            .selected
            .try_into()
            .ok()
            .filter(|&s| s < self.data.len());

        result
    }

    /// Modifies the data array by applying a sampler to it.
    ///
    /// # Panics
    ///
    /// Panics if the vendored sampler throws a C++ exception. `llama_sampler_apply` is
    /// documented to be a pure logit transform and is not expected to throw; if it does
    /// the failure is propagated as a panic per the crash-fast invariant.
    pub fn apply_sampler(&mut self, sampler: &LlamaSampler) {
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                let mut out_error: *mut std::os::raw::c_char = ptr::null_mut();
                let status = llama_cpp_bindings_sys::llama_rs_sampler_apply(
                    sampler.sampler,
                    c_llama_token_data_array,
                    &raw mut out_error,
                );
                if status != llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_OK {
                    let message = crate::ffi_error_reader::read_and_free_cpp_error(out_error);
                    panic!("llama_rs_sampler_apply returned status {status}: {message}");
                }
            });
        }
    }

    /// Modifies the data array by applying a sampler to it
    #[must_use]
    pub fn with_sampler(mut self, sampler: &mut LlamaSampler) -> Self {
        self.apply_sampler(sampler);
        self
    }

    /// Randomly selects a token from the candidates based on their probabilities.
    ///
    /// # Errors
    /// Returns [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token(&mut self, seed: u32) -> Result<LlamaToken, TokenSamplingError> {
        self.apply_sampler(&LlamaSampler::dist(seed));
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }

    /// Selects the token with the highest probability.
    ///
    /// # Errors
    /// Returns [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token_greedy(&mut self) -> Result<LlamaToken, TokenSamplingError> {
        self.apply_sampler(&LlamaSampler::greedy());
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }
}

#[cfg(test)]
mod tests {
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;

    use super::LlamaTokenDataArray;

    #[test]
    fn apply_greedy_sampler_selects_highest_logit() {
        use crate::sampling::LlamaSampler;

        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
            ],
            false,
        );

        array.apply_sampler(&LlamaSampler::greedy());

        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn with_sampler_builder_pattern() {
        use crate::sampling::LlamaSampler;

        let array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            ],
            false,
        )
        .with_sampler(&mut LlamaSampler::greedy());

        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn sample_token_greedy_returns_highest() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(10), 0.1, 0.0),
                LlamaTokenData::new(LlamaToken::new(20), 9.9, 0.0),
            ],
            false,
        );

        let token = array
            .sample_token_greedy()
            .expect("test: greedy sampler should select a token");

        assert_eq!(token, LlamaToken::new(20));
    }

    #[test]
    fn from_iter_creates_array_from_iterator() {
        let array = LlamaTokenDataArray::from_iter(
            [
                LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 2.0, 0.0),
            ],
            false,
        );

        assert_eq!(array.data.len(), 3);
        assert!(!array.sorted);
        assert!(array.selected.is_none());
    }

    #[test]
    fn sample_token_with_seed_selects_a_token() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(10), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(20), 1.0, 0.0),
            ],
            false,
        );

        let token = array
            .sample_token(42)
            .expect("test: dist sampler should select a token");

        assert!(token == LlamaToken::new(10) || token == LlamaToken::new(20));
    }

    #[test]
    fn selected_token_returns_none_when_no_selection() {
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        assert!(array.selected_token().is_none());
    }

    #[test]
    fn selected_token_returns_none_when_index_out_of_bounds() {
        let array = LlamaTokenDataArray {
            data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            selected: Some(5),
            sorted: false,
        };

        assert!(array.selected_token().is_none());
    }

    #[test]
    fn modify_as_c_llama_token_data_array_copies_when_data_pointer_changes() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
            ],
            false,
        );

        let replacement = [
            llama_cpp_bindings_sys::llama_token_data {
                id: 10,
                logit: 5.0,
                p: 0.0,
            },
            llama_cpp_bindings_sys::llama_token_data {
                id: 20,
                logit: 6.0,
                p: 0.0,
            },
        ];

        unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                c_array.data = replacement.as_ptr().cast_mut();
                c_array.size = replacement.len();
                c_array.selected = 0;
            });
        }

        assert_eq!(array.data.len(), 2);
        assert_eq!(array.data[0].id(), LlamaToken::new(10));
        assert_eq!(array.data[1].id(), LlamaToken::new(20));
        assert_eq!(array.selected, Some(0));
    }

    #[test]
    fn selected_overflow_uses_negative_one() {
        let mut array = LlamaTokenDataArray {
            data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            selected: Some(usize::MAX),
            sorted: false,
        };

        unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                assert_eq!(c_array.selected, -1);
            });
        }
    }
}
