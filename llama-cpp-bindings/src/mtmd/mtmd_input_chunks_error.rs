/// Errors that can occur when working with MTMD input chunks collections
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// Input chunks creation returned null
    #[error("Input chunks creation returned null")]
    NullResult,
}
