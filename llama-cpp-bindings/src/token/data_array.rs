use std::ptr;

use crate::error::SamplerApplyError;
use crate::error::TokenSamplingError;
use crate::sampling::LlamaSampler;
use crate::token::data::LlamaTokenData;

use super::LlamaToken;

fn sampler_apply_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_apply_status,
    out_error: *mut std::os::raw::c_char,
) -> Result<(), SamplerApplyError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_NULL_SAMPLER_ARG => {
            Err(SamplerApplyError::NullSampler)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplerApplyError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(SamplerApplyError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_sampler_apply returned unrecognized status {other}")
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenDataArray {
    pub data: Vec<LlamaTokenData>,
    pub selected: Option<usize>,
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    #[must_use]
    pub const fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    pub fn from_iter<TIterator>(data: TIterator, sorted: bool) -> Self
    where
        TIterator: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }
}

impl LlamaTokenDataArray {
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

    /// # Errors
    ///
    /// Returns [`SamplerApplyError`] if the sampler pointer is null, the vendored
    /// sampler runs out of memory, or it throws a C++ exception while applying.
    pub fn apply_sampler(&mut self, sampler: &LlamaSampler) -> Result<(), SamplerApplyError> {
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                let mut out_error: *mut std::os::raw::c_char = ptr::null_mut();
                let status = llama_cpp_bindings_sys::llama_rs_sampler_apply(
                    sampler.sampler,
                    c_llama_token_data_array,
                    &raw mut out_error,
                );
                sampler_apply_status_to_result(status, out_error)
            })
        }
    }

    /// # Errors
    /// Returns [`SamplerApplyError`] if applying the sampler fails.
    pub fn with_sampler(mut self, sampler: &mut LlamaSampler) -> Result<Self, SamplerApplyError> {
        self.apply_sampler(sampler)?;
        Ok(self)
    }

    /// # Errors
    /// Returns [`TokenSamplingError::SamplerApply`] if applying the sampler fails, or
    /// [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token(&mut self, seed: u32) -> Result<LlamaToken, TokenSamplingError> {
        self.apply_sampler(&LlamaSampler::dist(seed))?;
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }

    /// # Errors
    /// Returns [`TokenSamplingError::SamplerApply`] if applying the sampler fails, or
    /// [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token_greedy(&mut self) -> Result<LlamaToken, TokenSamplingError> {
        self.apply_sampler(&LlamaSampler::greedy())?;
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::SamplerApplyError;
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;

    use super::LlamaTokenDataArray;
    use super::sampler_apply_status_to_result;

    #[test]
    fn sampler_apply_status_allocation_failed_returns_not_enough_memory() {
        assert_eq!(
            sampler_apply_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED,
                std::ptr::null_mut(),
            ),
            Err(SamplerApplyError::NotEnoughMemory),
        );
    }

    #[test]
    fn sampler_apply_status_cxx_exception_returns_reported_with_unknown_message() {
        assert_eq!(
            sampler_apply_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_VENDORED_THREW_CXX_EXCEPTION,
                std::ptr::null_mut(),
            ),
            Err(SamplerApplyError::Reported {
                message: "unknown error".to_owned(),
            }),
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_sampler_apply returned unrecognized status")]
    fn sampler_apply_status_unrecognized_panics() {
        let _ = sampler_apply_status_to_result(u32::MAX, std::ptr::null_mut());
    }

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

        array
            .apply_sampler(&LlamaSampler::greedy())
            .expect("test: greedy sampler must apply");

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
        .with_sampler(&mut LlamaSampler::greedy())
        .expect("test: building with greedy sampler must succeed");

        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn with_sampler_with_null_sampler_returns_sampler_apply_error() {
        use crate::sampling::LlamaSampler;

        let mut null_sampler = LlamaSampler {
            sampler: std::ptr::null_mut(),
        };
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        assert_eq!(
            array.with_sampler(&mut null_sampler),
            Err(SamplerApplyError::NullSampler),
        );
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
    fn apply_sampler_with_null_sampler_returns_null_sampler_error() {
        use crate::sampling::LlamaSampler;

        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        let null_sampler = LlamaSampler {
            sampler: std::ptr::null_mut(),
        };

        assert_eq!(
            array.apply_sampler(&null_sampler),
            Err(SamplerApplyError::NullSampler)
        );
    }

    #[test]
    fn modify_clears_selection_when_index_is_out_of_range() {
        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                c_array.selected = 5;
            });
        }

        assert_eq!(array.selected, None);
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

    #[test]
    fn preset_valid_selection_is_passed_through_as_index() {
        let mut array = LlamaTokenDataArray {
            data: vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
            ],
            selected: Some(1),
            sorted: false,
        };

        unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                assert_eq!(c_array.selected, 1);
            });
        }

        assert_eq!(array.selected, Some(1));
    }
}
