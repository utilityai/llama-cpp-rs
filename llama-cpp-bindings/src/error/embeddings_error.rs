#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
    #[error("Can't use sequence embeddings with a model supporting only LLAMA_POOLING_TYPE_NONE")]
    NonePoolType,
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(#[source] std::num::TryFromIntError),
}
