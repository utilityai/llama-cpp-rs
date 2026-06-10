use crate::error::sampler_apply_error::SamplerApplyError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenSamplingError {
    #[error("No token was selected by the sampler")]
    NoTokenSelected,
    #[error("applying the sampler to the token data array failed: {0}")]
    SamplerApply(#[from] SamplerApplyError),
}
