/// Returned by [`crate::model::params::LlamaModelParams::fit_params`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    /// Vendored `common_fit_params` reported FAILURE: no allocation that fits available memory was found.
    #[error("common_fit_params reported FAILURE: no allocations that fit available memory")]
    VendoredReportedFailure,
    /// Vendored `common_fit_params` reported ERROR: a hard error occurred during fitting (e.g. model file not found).
    #[error("common_fit_params reported ERROR: hard error during parameter fitting")]
    VendoredReportedError,
    /// Vendored `common_fit_params` returned a status code the wrapper does not recognise.
    #[error("common_fit_params returned an unrecognised status code: {code}")]
    VendoredReturnedUnrecognizedStatusCode { code: i32 },
    /// Wrapper failed to duplicate the C++ exception message into a Rust-owned string.
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    /// Vendored `common_fit_params` threw a C++ exception caught at the wrapper boundary.
    #[error("common_fit_params threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
