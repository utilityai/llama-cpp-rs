/// Errors that can occur when initializing MTMD context
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Failed to create `CString` from input
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// MTMD context initialization returned null
    #[error("MTMD context initialization returned null")]
    NullResult,
}
