/// Errors that can occur during encoding
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// Encode operation failed
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}
