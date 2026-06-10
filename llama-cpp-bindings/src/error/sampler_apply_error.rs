#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SamplerApplyError {
    #[error("the sampler pointer was null when applying to the token data array")]
    NullSampler,
    #[error("the sampler ran out of memory while applying to the token data array")]
    NotEnoughMemory,
    #[error(
        "the vendored sampler threw a C++ exception while applying to the token data array: {message}"
    )]
    Reported { message: String },
}
