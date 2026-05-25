use crate::mtmd::MtmdEvalError;
use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

#[derive(Debug, thiserror::Error)]
pub enum EvalMultimodalChunksError {
    #[error("{0}")]
    EvalFailed(#[from] MtmdEvalError),
    #[error("{0}")]
    UnknownChunkType(#[from] MtmdInputChunkTypeError),
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
}
