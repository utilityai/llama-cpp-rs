/// Failed to accept a token in a sampler.
#[derive(Debug, thiserror::Error)]
pub enum SamplerAcceptError {
    /// A C++ exception was thrown during accept
    #[error("C++ exception during sampler accept: {0}")]
    CppException(String),

    /// An invalid argument was passed (null sampler or null error pointer)
    #[error("Invalid argument passed to sampler accept")]
    InvalidArgument,
}
