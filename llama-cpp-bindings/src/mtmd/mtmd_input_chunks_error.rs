#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdInputChunksError {
    #[error("input chunks collection could not be created")]
    ChunksCreationFailed,
}
