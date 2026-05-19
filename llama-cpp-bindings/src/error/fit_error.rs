/// Returned by [`crate::model::params::LlamaModelParams::fit_params`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    /// No combination of model parameters fits the available device memory.
    #[error("no parameter combination fits available memory")]
    NoFittingMemoryLayout,
    /// Parameter fitting was aborted by a hard error reported by the underlying library
    /// (e.g., model file missing, backend initialization failed).
    #[error("parameter fitting aborted")]
    Aborted,
    /// The fitting helper returned a status code the wrapper does not recognise.
    #[error("parameter fitting returned an unknown status code: {code}")]
    UnknownStatus { code: i32 },
    /// Wrapper could not allocate memory for an error message.
    #[error("not enough memory")]
    NotEnoughMemory,
    /// Generic exception caught at the wrapper boundary, with the underlying message.
    #[error("{message}")]
    Reported { message: String },
}
