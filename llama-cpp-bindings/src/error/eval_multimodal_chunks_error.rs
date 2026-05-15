use crate::mtmd::MtmdEvalError;
use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

/// Failed to evaluate multimodal chunks through the request classifier.
#[derive(Debug, thiserror::Error)]
pub enum EvalMultimodalChunksError {
    /// `MtmdInputChunks::eval_chunks` returned an error.
    #[error("{0}")]
    EvalFailed(#[from] MtmdEvalError),
    /// A chunk reported a type that is not known to this binding.
    #[error("{0}")]
    UnknownChunkType(#[from] MtmdInputChunkTypeError),
    /// A chunk index that was within `chunks.len()` returned `None` from `chunks.get(index)`.
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
}
