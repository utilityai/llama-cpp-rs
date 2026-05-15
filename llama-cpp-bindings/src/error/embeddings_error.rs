/// When embedding related functions fail
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    /// Embeddings weren't enabled in the context options
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    /// Logits weren't enabled for the given token
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
    /// The given sequence index exceeds the max sequence id
    #[error("Can't use sequence embeddings with a model supporting only LLAMA_POOLING_TYPE_NONE")]
    NonePoolType,
    /// The embedding dimension does not fit into a usize.
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(#[source] std::num::TryFromIntError),
}
