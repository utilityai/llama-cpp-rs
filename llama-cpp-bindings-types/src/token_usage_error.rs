#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenUsageError {
    #[error(
        "cached prompt tokens would reach {cached_after} but only {prompt} prompt tokens were recorded"
    )]
    CachedExceedsPrompt {
        cached_after: u64,
        prompt: u64,
    },
}
