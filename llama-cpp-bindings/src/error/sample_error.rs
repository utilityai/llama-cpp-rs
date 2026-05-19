#[derive(Debug, thiserror::Error)]
pub enum SampleError {
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
