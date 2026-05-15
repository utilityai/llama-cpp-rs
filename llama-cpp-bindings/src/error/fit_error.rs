/// Returned by [`crate::model::params::LlamaModelParams::fit_params`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum FitError {
    /// Could not find allocations that fit available memory.
    #[error("could not find allocations that fit available memory")]
    Failure,
    /// A hard error occurred during fitting (e.g. model not found at the specified path,
    /// or the C++ wrapper threw an exception).
    #[error("hard error during parameter fitting")]
    Error,
}
