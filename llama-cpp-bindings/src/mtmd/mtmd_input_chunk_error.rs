/// Errors that can occur when working with individual MTMD input chunks
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// Input chunk operation returned null
    #[error("Input chunk operation returned null")]
    NullResult,
}
