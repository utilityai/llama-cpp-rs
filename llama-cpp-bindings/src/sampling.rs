use std::borrow::Borrow;
use std::ffi::{CString, c_char};
use std::fmt::{Debug, Formatter};

use llama_cpp_error_recorder::error_scope::ErrorScope;
use llama_cpp_error_recorder::recorded_error::RecordedError;
use llama_cpp_gbnf::gbnf_grammar::GbnfGrammar;
use llama_cpp_gbnf::gbnf_parse_error::GbnfParseError;

use crate::context::LlamaContext;
use crate::error::grammar_error::GrammarError;
use crate::error::sample_error::SampleError;
use crate::error::sampler_accept_error::SamplerAcceptError;
use crate::error::sampling_error::SamplingError;
use crate::ffi_error_reader::read_and_free_cpp_error;
use crate::model::LlamaModel;
use crate::token::LlamaToken;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::logit_bias::LlamaLogitBias;

fn check_sampler_accept_status(
    status: llama_cpp_bindings_sys::llama_rs_sampler_accept_status,
    error_ptr: *mut c_char,
) -> Result<(), SamplerAcceptError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplerAcceptError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(SamplerAcceptError::GrammarStateCorrupted { message })
        }
        other => Err(SamplerAcceptError::UnrecognizedStatusCode { code: other }),
    }
}

fn sampler_sample_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_sample_status,
    token: i32,
    error_ptr: *mut c_char,
) -> Result<LlamaToken, SampleError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_OK => Ok(LlamaToken(token)),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SampleError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(SampleError::Reported { message })
        }
        other => Err(SampleError::UnrecognizedStatusCode { code: other }),
    }
}

fn sampler_init_grammar_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
) -> Result<LlamaSampler, GrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_OK => Ok(LlamaSampler { sampler }),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_COMPILATION_FAILED => {
            Err(GrammarError::GrammarMalformed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(GrammarError::Reported { message })
        }
        other => Err(GrammarError::UnrecognizedStatusCode { code: other }),
    }
}

fn sampler_init_grammar_lazy_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
) -> Result<LlamaSampler, GrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_OK => {
            Ok(LlamaSampler { sampler })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_COMPILATION_FAILED => {
            Err(GrammarError::LazyGrammarMalformed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(GrammarError::Reported { message })
        }
        other => {
            Err(GrammarError::UnrecognizedStatusCode { code: other })
        }
    }
}

fn sampler_init_grammar_lazy_patterns_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy_patterns_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
) -> Result<LlamaSampler, GrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_OK => {
            Ok(LlamaSampler { sampler })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_COMPILATION_FAILED => {
            Err(GrammarError::LazyPatternsGrammarMalformed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_INVALID_TRIGGER_PATTERN => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(GrammarError::InvalidTriggerPattern { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_error(error_ptr) };
            Err(GrammarError::Reported { message })
        }
        other => Err(GrammarError::UnrecognizedStatusCode { code: other }),
    }
}

fn n_ctx_train_overflow_to_grammar_error(convert_error: std::num::TryFromIntError) -> GrammarError {
    GrammarError::IntegerOverflow(format!(
        "n_ctx_train does not fit into u32: {convert_error}"
    ))
}

fn checked_u32_as_i32(value: u32) -> Result<i32, GrammarError> {
    i32::try_from(value).map_err(|convert_error| {
        GrammarError::IntegerOverflow(format!("value exceeds i32::MAX: {convert_error}"))
    })
}

fn checked_usize_as_i32_sampling(value: usize) -> Result<i32, SamplingError> {
    i32::try_from(value).map_err(|convert_error| {
        SamplingError::IntegerOverflow(format!("value exceeds i32::MAX: {convert_error}"))
    })
}

pub struct LlamaSampler {
    pub sampler: *mut llama_cpp_bindings_sys::llama_sampler,
}

fn grammar_callback_error_to_result(error: Option<RecordedError>) -> Result<(), SampleError> {
    error.map_or(Ok(()), |recorded| {
        Err(SampleError::GrammarCallbackFailed {
            message: recorded.into_message(),
        })
    })
}

fn grammar_callback_error_to_accept_result(
    error: Option<RecordedError>,
) -> Result<(), SamplerAcceptError> {
    error.map_or(Ok(()), |recorded| {
        Err(SamplerAcceptError::GrammarCallbackFailed {
            message: recorded.into_message(),
        })
    })
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChain").finish()
    }
}

impl LlamaSampler {
    /// # Errors
    ///
    /// Returns [`SampleError`] if the C++ sampler throws an exception, the index is invalid, or the
    /// grammar sampler callback recorded a failure during sampling.
    pub fn sample(&mut self, ctx: &LlamaContext, idx: i32) -> Result<LlamaToken, SampleError> {
        let mut token: i32 = -1;
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let scope = ErrorScope::enter();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_sample(
                self.sampler,
                ctx.context.as_ptr(),
                idx,
                &raw mut token,
                &raw mut error_ptr,
            )
        };
        grammar_callback_error_to_result(scope.take())?;

        sampler_sample_status_to_result(status, token, error_ptr)
    }

    /// # Errors
    ///
    /// Returns [`SampleError`] if the grammar sampler callback recorded a failure during application.
    pub fn apply(&self, data_array: &mut LlamaTokenDataArray) -> Result<(), SampleError> {
        let scope = ErrorScope::enter();
        data_array.apply_sampler(self)?;

        grammar_callback_error_to_result(scope.take())
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects the token.
    pub fn accept(&mut self, token: LlamaToken) -> Result<(), SamplerAcceptError> {
        self.try_accept(token)
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects any token.
    pub fn accept_many(
        &mut self,
        tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>,
    ) -> Result<(), SamplerAcceptError> {
        for token in tokens {
            self.try_accept(*token.borrow())?;
        }

        Ok(())
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects any token.
    pub fn with_tokens(
        mut self,
        tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>,
    ) -> Result<Self, SamplerAcceptError> {
        self.accept_many(tokens)?;

        Ok(self)
    }

    /// # Errors
    /// Returns an error if the underlying sampler rejects the token.
    pub fn try_accept(&mut self, token: LlamaToken) -> Result<(), SamplerAcceptError> {
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let scope = ErrorScope::enter();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_accept(
                self.sampler,
                token.0,
                &raw mut error_ptr,
            )
        };
        grammar_callback_error_to_accept_result(scope.take())?;

        check_sampler_accept_status(status, error_ptr)
    }

    /// # Errors
    ///
    /// Returns [`SampleError`] if the grammar sampler callback recorded a failure during reset.
    pub fn reset(&mut self) -> Result<(), SampleError> {
        let scope = ErrorScope::enter();
        unsafe {
            llama_cpp_bindings_sys::llama_sampler_reset(self.sampler);
        }

        grammar_callback_error_to_result(scope.take())
    }

    #[must_use]
    pub fn get_seed(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_sampler_get_seed(self.sampler) }
    }

    #[must_use]
    pub fn chain(samplers: impl IntoIterator<Item = Self>, no_perf: bool) -> Self {
        unsafe {
            let chain = llama_cpp_bindings_sys::llama_sampler_chain_init(
                llama_cpp_bindings_sys::llama_sampler_chain_params { no_perf },
            );

            for sampler in samplers {
                llama_cpp_bindings_sys::llama_sampler_chain_add(chain, sampler.sampler);
                std::mem::forget(sampler);
            }

            Self { sampler: chain }
        }
    }

    #[must_use]
    pub fn chain_simple(samplers: impl IntoIterator<Item = Self>) -> Self {
        Self::chain(samplers, false)
    }

    #[must_use]
    pub fn temp(t: f32) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_temp(t) };
        Self { sampler }
    }

    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Self {
        let sampler =
            unsafe { llama_cpp_bindings_sys::llama_sampler_init_temp_ext(t, delta, exponent) };
        Self { sampler }
    }

    #[must_use]
    pub fn top_k(k: i32) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_top_k(k) };
        Self { sampler }
    }

    #[must_use]
    pub fn top_n_sigma(n: f32) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_top_n_sigma(n) };
        Self { sampler }
    }

    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_typical(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_top_p(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_min_p(p, min_keep) };
        Self { sampler }
    }

    #[must_use]
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        let sampler =
            unsafe { llama_cpp_bindings_sys::llama_sampler_init_xtc(p, t, min_keep, seed) };
        Self { sampler }
    }

    /// # Errors
    /// Returns an error if the grammar is invalid or the sampler cannot be initialized.
    pub fn grammar(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<Self, GrammarError> {
        let (grammar_str, grammar_root) =
            Self::sanitize_grammar_strings(grammar_str, grammar_root)?;
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_grammar(
                model.vocab_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
                &raw mut sampler,
                &raw mut error_ptr,
            )
        };

        sampler_init_grammar_status_to_result(status, sampler, error_ptr)
    }

    /// # Errors
    /// Returns an error if the grammar or trigger words are invalid.
    pub fn grammar_lazy(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        trigger_words: impl IntoIterator<Item = impl AsRef<[u8]>>,
        trigger_tokens: &[LlamaToken],
    ) -> Result<Self, GrammarError> {
        let (grammar_str, grammar_root) =
            Self::sanitize_grammar_strings(grammar_str, grammar_root)?;
        let trigger_words = Self::sanitize_trigger_words(trigger_words)?;
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let mut trigger_word_ptrs: Vec<*const c_char> =
            trigger_words.iter().map(|cs| cs.as_ptr()).collect();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy(
                model.vocab_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
                trigger_word_ptrs.as_mut_ptr(),
                trigger_word_ptrs.len(),
                trigger_tokens.as_ptr().cast(),
                trigger_tokens.len(),
                &raw mut sampler,
                &raw mut error_ptr,
            )
        };

        sampler_init_grammar_lazy_status_to_result(status, sampler, error_ptr)
    }

    /// # Errors
    /// Returns an error if the grammar or trigger patterns are invalid.
    pub fn grammar_lazy_patterns(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        trigger_patterns: &[String],
        trigger_tokens: &[LlamaToken],
    ) -> Result<Self, GrammarError> {
        let (grammar_str, grammar_root) =
            Self::sanitize_grammar_strings(grammar_str, grammar_root)?;
        let trigger_patterns = Self::sanitize_trigger_patterns(trigger_patterns)?;
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let mut trigger_pattern_ptrs: Vec<*const c_char> =
            trigger_patterns.iter().map(|cs| cs.as_ptr()).collect();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy_patterns(
                model.vocab_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
                trigger_pattern_ptrs.as_mut_ptr(),
                trigger_pattern_ptrs.len(),
                trigger_tokens.as_ptr().cast(),
                trigger_tokens.len(),
                &raw mut sampler,
                &raw mut error_ptr,
            )
        };

        sampler_init_grammar_lazy_patterns_status_to_result(status, sampler, error_ptr)
    }

    /// # Errors
    ///
    /// Returns [`GrammarError`] if the grammar is invalid or the sampler cannot be initialized.
    pub fn llguidance(
        model: &LlamaModel,
        grammar_kind: &str,
        grammar_data: &str,
    ) -> Result<Self, GrammarError> {
        crate::llguidance_sampler::create_llg_sampler(model, grammar_kind, grammar_data)
    }

    fn sanitize_grammar_strings(
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<(CString, CString), GrammarError> {
        if let Err(GbnfParseError::RootSymbolMissing { .. }) =
            GbnfGrammar::parse(grammar_str, grammar_root)
        {
            return Err(GrammarError::RootNotFound);
        }

        let grammar = CString::new(grammar_str).map_err(GrammarError::GrammarNullBytes)?;
        let root = CString::new(grammar_root).map_err(GrammarError::GrammarNullBytes)?;

        Ok((grammar, root))
    }

    fn sanitize_trigger_words(
        trigger_words: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Result<Vec<CString>, GrammarError> {
        trigger_words
            .into_iter()
            .map(|word| CString::new(word.as_ref()).map_err(GrammarError::TriggerWordNullBytes))
            .collect()
    }

    fn sanitize_trigger_patterns(
        trigger_patterns: &[String],
    ) -> Result<Vec<CString>, GrammarError> {
        trigger_patterns
            .iter()
            .map(|pattern| CString::new(pattern.as_str()).map_err(GrammarError::GrammarNullBytes))
            .collect()
    }

    /// # Errors
    /// Returns an error if any string in `seq_breakers` contains null bytes.
    pub fn dry(
        model: &LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Result<Self, GrammarError> {
        let seq_breakers: Vec<CString> = seq_breakers
            .into_iter()
            .map(|seq_breaker| CString::new(seq_breaker.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seq_breaker_pointers: Vec<*const c_char> = seq_breakers
            .iter()
            .map(|seq_breaker| seq_breaker.as_ptr())
            .collect();

        let n_ctx_train_value = model
            .n_ctx_train()
            .map_err(n_ctx_train_overflow_to_grammar_error)?;
        let n_ctx_train = checked_u32_as_i32(n_ctx_train_value)?;
        let sampler = unsafe {
            llama_cpp_bindings_sys::llama_sampler_init_dry(
                model.vocab_ptr(),
                n_ctx_train,
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                seq_breaker_pointers.as_mut_ptr(),
                seq_breaker_pointers.len(),
            )
        };

        Ok(Self { sampler })
    }

    #[must_use]
    pub fn penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        let sampler = unsafe {
            llama_cpp_bindings_sys::llama_sampler_init_penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            )
        };
        Self { sampler }
    }

    #[must_use]
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        let sampler = unsafe {
            llama_cpp_bindings_sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
        };
        Self { sampler }
    }

    #[must_use]
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        let sampler =
            unsafe { llama_cpp_bindings_sys::llama_sampler_init_mirostat_v2(seed, tau, eta) };
        Self { sampler }
    }

    #[must_use]
    pub fn dist(seed: u32) -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_dist(seed) };
        Self { sampler }
    }

    #[must_use]
    pub fn greedy() -> Self {
        let sampler = unsafe { llama_cpp_bindings_sys::llama_sampler_init_greedy() };
        Self { sampler }
    }

    /// # Errors
    /// Returns [`SamplingError::IntegerOverflow`] if `biases.len()` exceeds `i32::MAX`.
    ///
    pub fn logit_bias(n_vocab: i32, biases: &[LlamaLogitBias]) -> Result<Self, SamplingError> {
        let bias_count = checked_usize_as_i32_sampling(biases.len())?;
        let data = biases
            .as_ptr()
            .cast::<llama_cpp_bindings_sys::llama_logit_bias>();

        let sampler = unsafe {
            llama_cpp_bindings_sys::llama_sampler_init_logit_bias(n_vocab, bias_count, data)
        };

        Ok(Self { sampler })
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_sampler_free(self.sampler);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::mem::Discriminant;

    use llama_cpp_error_recorder::recorded_error::RecordedError;

    use super::LlamaSampler;
    use super::grammar_callback_error_to_accept_result;
    use super::grammar_callback_error_to_result;
    use crate::error::grammar_error::GrammarError;
    use crate::error::sample_error::SampleError;
    use crate::error::sampler_accept_error::SamplerAcceptError;

    #[test]
    fn grammar_callback_error_to_result_maps_recorded_error() {
        let result =
            grammar_callback_error_to_result(Some(RecordedError::new("mask failed".to_string())));

        assert_eq!(
            result.unwrap_err(),
            SampleError::GrammarCallbackFailed {
                message: "mask failed".to_string()
            }
        );
    }

    #[test]
    fn grammar_callback_error_to_result_maps_absence_to_ok() {
        assert!(grammar_callback_error_to_result(None).is_ok());
    }

    #[test]
    fn grammar_callback_error_to_accept_result_maps_recorded_error() {
        let result = grammar_callback_error_to_accept_result(Some(RecordedError::new(
            "consume failed".to_string(),
        )));

        assert_eq!(
            result,
            Err(SamplerAcceptError::GrammarCallbackFailed {
                message: "consume failed".to_string()
            })
        );
    }

    #[test]
    fn grammar_callback_error_to_accept_result_maps_absence_to_ok() {
        assert!(grammar_callback_error_to_accept_result(None).is_ok());
    }

    fn nul_error() -> std::ffi::NulError {
        CString::new(b"a\0b".to_vec()).unwrap_err()
    }

    fn root_not_found_disc() -> Discriminant<GrammarError> {
        std::mem::discriminant(&GrammarError::RootNotFound)
    }

    fn grammar_null_bytes_disc() -> Discriminant<GrammarError> {
        std::mem::discriminant(&GrammarError::GrammarNullBytes(nul_error()))
    }

    fn trigger_word_null_bytes_disc() -> Discriminant<GrammarError> {
        std::mem::discriminant(&GrammarError::TriggerWordNullBytes(nul_error()))
    }

    #[test]
    fn sanitize_grammar_strings_valid() {
        let result = LlamaSampler::sanitize_grammar_strings("root ::= \"hello\"", "root");

        assert!(result.is_ok());
    }

    #[test]
    fn sanitize_grammar_strings_root_not_found() {
        let err = LlamaSampler::sanitize_grammar_strings("expr ::= \"hello\"", "root").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), root_not_found_disc());
    }

    #[test]
    fn sanitize_grammar_strings_null_byte_in_grammar() {
        let err = LlamaSampler::sanitize_grammar_strings("root ::= \"\0\"", "root").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), grammar_null_bytes_disc());
    }

    #[test]
    fn sanitize_grammar_strings_null_byte_in_root() {
        let err =
            LlamaSampler::sanitize_grammar_strings("ro\0ot ::= \"hello\"", "ro\0ot").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), grammar_null_bytes_disc());
    }

    #[test]
    fn sanitize_trigger_words_valid() {
        let words: Vec<&[u8]> = vec![b"hello", b"world"];
        let result = LlamaSampler::sanitize_trigger_words(words);

        assert!(result.is_ok());
        assert_eq!(result.expect("valid trigger words").len(), 2);
    }

    #[test]
    fn sanitize_trigger_words_empty_list() {
        let words: Vec<&[u8]> = vec![];
        let result = LlamaSampler::sanitize_trigger_words(words);

        assert!(result.is_ok());
        assert!(result.expect("valid trigger words").is_empty());
    }

    #[test]
    fn sanitize_trigger_words_null_byte() {
        let words: Vec<&[u8]> = vec![b"hel\0lo"];
        let err = LlamaSampler::sanitize_trigger_words(words).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), trigger_word_null_bytes_disc());
    }

    #[test]
    fn sanitize_trigger_patterns_valid() {
        let patterns = vec!["^hello$".to_string(), "world.*".to_string()];
        let result = LlamaSampler::sanitize_trigger_patterns(&patterns);

        assert!(result.is_ok());
        assert_eq!(result.expect("valid trigger patterns").len(), 2);
    }

    #[test]
    fn sanitize_trigger_patterns_empty_list() {
        let patterns: Vec<String> = vec![];
        let result = LlamaSampler::sanitize_trigger_patterns(&patterns);

        assert!(result.is_ok());
        assert!(result.expect("valid trigger patterns").is_empty());
    }

    #[test]
    fn sanitize_trigger_patterns_null_byte() {
        let patterns = vec!["hel\0lo".to_string()];
        let err = LlamaSampler::sanitize_trigger_patterns(&patterns).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), grammar_null_bytes_disc());
    }

    #[test]
    fn apply_modifies_data_array() {
        use crate::token::LlamaToken;
        use crate::token::data::LlamaTokenData;
        use crate::token::data_array::LlamaTokenDataArray;

        let sampler = LlamaSampler::greedy();
        let mut data_array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            ],
            false,
        );

        assert!(sampler.apply(&mut data_array).is_ok());

        assert_eq!(data_array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn apply_with_null_sampler_surfaces_sampler_apply_error() {
        use crate::error::sample_error::SampleError;
        use crate::error::sampler_apply_error::SamplerApplyError;
        use crate::token::LlamaToken;
        use crate::token::data::LlamaTokenData;
        use crate::token::data_array::LlamaTokenDataArray;

        let null_sampler = LlamaSampler {
            sampler: std::ptr::null_mut(),
        };
        let mut data_array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        assert_eq!(
            null_sampler.apply(&mut data_array),
            Err(SampleError::SamplerApply(SamplerApplyError::NullSampler)),
        );
    }

    #[test]
    fn accept_succeeds() {
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);

        sampler
            .accept(crate::token::LlamaToken::new(1))
            .expect("test: accept should succeed");
    }

    #[test]
    fn try_accept_succeeds_on_penalties_sampler() {
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);

        let result = sampler.try_accept(crate::token::LlamaToken::new(42));

        assert!(result.is_ok());
    }

    #[test]
    fn accept_many_multiple_tokens() {
        use crate::token::LlamaToken;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);

        sampler
            .accept_many([LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)])
            .expect("test: accept_many should succeed");
    }

    #[test]
    fn with_tokens_builder_pattern() {
        use crate::token::LlamaToken;

        let _sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
        .with_tokens([LlamaToken::new(10), LlamaToken::new(20)])
        .expect("test: with_tokens should succeed");
    }

    #[test]
    fn all_sampler_constructors() {
        use crate::token::LlamaToken;
        use crate::token::logit_bias::LlamaLogitBias;

        let _temp = LlamaSampler::temp(0.8);
        let _temp_ext = LlamaSampler::temp_ext(0.8, 0.1, 1.0);
        let _top_k = LlamaSampler::top_k(40);
        let _top_n_sigma = LlamaSampler::top_n_sigma(2.0);
        let _top_p = LlamaSampler::top_p(0.9, 1);
        let _min_p = LlamaSampler::min_p(0.05, 1);
        let _typical = LlamaSampler::typical(0.9, 1);
        let _xtc = LlamaSampler::xtc(0.1, 0.5, 1, 42);
        let _dist = LlamaSampler::dist(42);
        let _mirostat = LlamaSampler::mirostat(32000, 42, 5.0, 0.1, 100);
        let _mirostat_v2 = LlamaSampler::mirostat_v2(42, 5.0, 0.1);
        let biases = vec![LlamaLogitBias::new(LlamaToken::new(0), -100.0)];
        let _logit_bias = LlamaSampler::logit_bias(32000, &biases);
        let _chain = LlamaSampler::chain([LlamaSampler::greedy()], true);
    }

    #[test]
    fn reset_and_get_seed() {
        let mut sampler = LlamaSampler::dist(42);
        assert!(sampler.reset().is_ok());
        let _seed = sampler.get_seed();
    }

    #[test]
    fn debug_formatting() {
        let sampler = LlamaSampler::greedy();
        let debug_output = format!("{sampler:?}");
        assert!(debug_output.contains("LlamaSampler"));
    }

    #[test]
    fn checked_u32_as_i32_overflow() {
        let result = super::checked_u32_as_i32(u32::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn checked_usize_as_i32_sampling_overflow() {
        let result = super::checked_usize_as_i32_sampling(usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn check_sampler_accept_status_ok() {
        let result = super::check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_OK,
            std::ptr::null_mut(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn check_sampler_accept_status_exception_maps_to_typed_variant() {
        let err = super::check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
        )
        .unwrap_err();
        let grammar_state_corrupted_disc =
            std::mem::discriminant(&SamplerAcceptError::GrammarStateCorrupted {
                message: String::new(),
            });

        assert_eq!(std::mem::discriminant(&err), grammar_state_corrupted_disc);
    }

    #[test]
    fn check_sampler_accept_status_allocation_failure_maps_to_not_enough_memory() {
        let result = super::check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(SamplerAcceptError::NotEnoughMemory));
    }

    #[test]
    fn check_sampler_accept_status_unrecognized_returns_unrecognized_status_error() {
        assert_eq!(
            super::check_sampler_accept_status(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG,
                std::ptr::null_mut(),
            ),
            Err(SamplerAcceptError::UnrecognizedStatusCode {
                code: llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG
            }),
        );
    }

    #[test]
    fn sampler_sample_status_allocation_failure_maps_to_not_enough_memory() {
        let result = super::sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED,
            -1,
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), SampleError::NotEnoughMemory);
    }

    #[test]
    fn sampler_sample_status_exception_maps_to_reported() {
        let result = super::sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_THREW_CXX_EXCEPTION,
            -1,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            SampleError::Reported {
                message: "unknown error".to_string()
            }
        );
    }

    #[test]
    fn sampler_sample_status_unrecognized_returns_unrecognized_status_error() {
        assert_eq!(
            super::sampler_sample_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG,
                -1,
                std::ptr::null_mut(),
            ),
            Err(SampleError::UnrecognizedStatusCode {
                code: llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG
            }),
        );
    }

    #[test]
    fn sampler_init_grammar_status_null_maps_to_grammar_malformed() {
        let result = super::sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_COMPILATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::GrammarMalformed);
    }

    #[test]
    fn sampler_init_grammar_status_allocation_failure_maps_to_not_enough_memory() {
        let result = super::sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::NotEnoughMemory);
    }

    #[test]
    fn sampler_init_grammar_status_exception_maps_to_reported() {
        let result = super::sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            GrammarError::Reported {
                message: "unknown error".to_string()
            }
        );
    }

    #[test]
    fn sampler_init_grammar_status_unrecognized_returns_unrecognized_status_error() {
        assert!(matches!(
            super::sampler_init_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            Err(GrammarError::UnrecognizedStatusCode { .. })
        ));
    }

    #[test]
    fn sampler_init_grammar_lazy_status_null_maps_to_lazy_grammar_malformed() {
        let result = super::sampler_init_grammar_lazy_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_COMPILATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::LazyGrammarMalformed);
    }

    #[test]
    fn sampler_init_grammar_lazy_status_allocation_failure_maps_to_not_enough_memory() {
        let result = super::sampler_init_grammar_lazy_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::NotEnoughMemory);
    }

    #[test]
    fn sampler_init_grammar_lazy_status_exception_maps_to_reported() {
        let result = super::sampler_init_grammar_lazy_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            GrammarError::Reported {
                message: "unknown error".to_string()
            }
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_status_unrecognized_returns_unrecognized_status_error() {
        assert!(matches!(
            super::sampler_init_grammar_lazy_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_SAMPLER_ARG,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            Err(GrammarError::UnrecognizedStatusCode { .. })
        ));
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_null_maps_to_lazy_patterns_grammar_malformed() {
        let result = super::sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_COMPILATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            GrammarError::LazyPatternsGrammarMalformed
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_allocation_failure_maps_to_not_enough_memory() {
        let result = super::sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::NotEnoughMemory);
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_exception_maps_to_reported() {
        let result = super::sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            GrammarError::Reported {
                message: "unknown error".to_string()
            }
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_unrecognized_returns_error() {
        assert!(matches!(
            super::sampler_init_grammar_lazy_patterns_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            Err(GrammarError::UnrecognizedStatusCode { .. })
        ));
    }

    #[test]
    fn n_ctx_train_overflow_maps_to_integer_overflow() {
        let convert_error = u32::try_from(-1_i64).expect_err("-1 cannot convert to u32");
        let grammar_error = super::n_ctx_train_overflow_to_grammar_error(convert_error);

        assert_eq!(
            std::mem::discriminant(&grammar_error),
            std::mem::discriminant(&GrammarError::IntegerOverflow(String::new())),
        );
    }

    #[test]
    fn grammar_returns_root_not_found_before_touching_model() {
        let model = unsafe { &*std::ptr::NonNull::<crate::model::LlamaModel>::dangling().as_ptr() };

        let err = LlamaSampler::grammar(model, "expr ::= \"hello\"", "root").unwrap_err();

        assert_eq!(err, GrammarError::RootNotFound);
    }
}
