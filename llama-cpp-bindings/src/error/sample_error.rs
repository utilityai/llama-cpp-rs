/// Errors that can occur when sampling a token.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SampleError {
    /// A C++ exception was thrown during sampling
    #[error("C++ exception during sampling: {0}")]
    CppException(String),

    /// An invalid argument was passed to the sampler
    #[error("Invalid argument passed to sampler")]
    InvalidArgument,
}
