#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
}
