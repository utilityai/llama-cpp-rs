/// Failed to sample a token from the data array.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenSamplingError {
    /// The sampler did not select any token.
    #[error("No token was selected by the sampler")]
    NoTokenSelected,
}
