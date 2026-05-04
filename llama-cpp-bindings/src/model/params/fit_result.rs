/// Result of [`crate::model::params::LlamaModelParams::fit_params`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FitResult {
    /// The context size after fitting (may have been reduced from the requested value).
    pub n_ctx: u32,
}
