/// Errors that can occur when creating a sampling configuration.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    /// An integer value exceeded the allowed range
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
}
