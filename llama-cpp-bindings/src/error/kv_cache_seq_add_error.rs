use std::num::TryFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheSeqAddError {
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("model rope type is incompatible with sequence position arithmetic")]
    IncompatibleRopeType,
    #[error("context has no memory module available")]
    MemoryHandleUnavailable,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
}
