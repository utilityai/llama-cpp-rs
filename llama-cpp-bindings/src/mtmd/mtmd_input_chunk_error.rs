#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    #[error("input chunk operation failed")]
    ChunkOperationFailed,
}
