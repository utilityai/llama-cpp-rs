#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenSamplingError {
    #[error("No token was selected by the sampler")]
    NoTokenSelected,
}
