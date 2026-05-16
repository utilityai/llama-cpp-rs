use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("grammar root not found in grammar string")]
    RootNotFound,
    #[error("trigger word contains null bytes: {0}")]
    TriggerWordNullBytes(NulError),
    #[error("grammar string or root contains null bytes: {0}")]
    GrammarNullBytes(NulError),
    #[error("string contains null bytes: {0}")]
    NulError(#[from] NulError),
    #[error("integer overflow: {0}")]
    IntegerOverflow(String),
    #[error("llguidance error: {0}")]
    LlguidanceError(String),
    #[error("llama_rs_sampler_init_grammar called with null out_sampler")]
    GrammarInitNullOutSamplerArg,
    #[error("llama_rs_sampler_init_grammar called with null out_error")]
    GrammarInitNullOutErrorArg,
    #[error("llama_rs_sampler_init_grammar returned null")]
    GrammarInitVendoredReturnedNull,
    #[error("llama_rs_sampler_init_grammar wrapper failed to duplicate the C++ exception string")]
    GrammarInitErrorStringAllocationFailed,
    #[error("llama_rs_sampler_init_grammar threw a C++ exception: {message}")]
    GrammarInitVendoredThrewCxxException { message: String },
    #[error("llama_rs_sampler_init_grammar_lazy called with null out_sampler")]
    GrammarLazyInitNullOutSamplerArg,
    #[error("llama_rs_sampler_init_grammar_lazy called with null out_error")]
    GrammarLazyInitNullOutErrorArg,
    #[error("llama_rs_sampler_init_grammar_lazy returned null")]
    GrammarLazyInitVendoredReturnedNull,
    #[error(
        "llama_rs_sampler_init_grammar_lazy wrapper failed to duplicate the C++ exception string"
    )]
    GrammarLazyInitErrorStringAllocationFailed,
    #[error("llama_rs_sampler_init_grammar_lazy threw a C++ exception: {message}")]
    GrammarLazyInitVendoredThrewCxxException { message: String },
    #[error("llama_rs_sampler_init_grammar_lazy_patterns called with null out_sampler")]
    GrammarLazyPatternsInitNullOutSamplerArg,
    #[error("llama_rs_sampler_init_grammar_lazy_patterns called with null out_error")]
    GrammarLazyPatternsInitNullOutErrorArg,
    #[error("llama_rs_sampler_init_grammar_lazy_patterns returned null")]
    GrammarLazyPatternsInitVendoredReturnedNull,
    #[error(
        "llama_rs_sampler_init_grammar_lazy_patterns wrapper failed to duplicate the C++ exception string"
    )]
    GrammarLazyPatternsInitErrorStringAllocationFailed,
    #[error("llama_rs_sampler_init_grammar_lazy_patterns threw a C++ exception: {message}")]
    GrammarLazyPatternsInitVendoredThrewCxxException { message: String },
    #[error("vendored llama_sampler_init for llguidance returned null")]
    LlguidanceSamplerInitVendoredReturnedNull,
}
