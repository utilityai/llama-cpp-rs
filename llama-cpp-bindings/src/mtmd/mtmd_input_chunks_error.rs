#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    #[error("input chunks collection could not be created")]
    ChunksCreationFailed,
}
