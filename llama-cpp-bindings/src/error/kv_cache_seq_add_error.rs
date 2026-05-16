use std::num::TryFromIntError;

#[derive(Debug, thiserror::Error)]
pub enum KvCacheSeqAddError {
    #[error("provided start position is too large for an i32")]
    P0TooLarge(#[source] TryFromIntError),
    #[error("provided end position is too large for an i32")]
    P1TooLarge(#[source] TryFromIntError),
    #[error("llama_rs_memory_seq_add called with null context")]
    NullContextArg,
    #[error("llama_rs_memory_seq_add invoked on a model with incompatible rope type")]
    IncompatibleRopeType,
    #[error("llama_rs_memory_seq_add could not acquire the context memory handle")]
    NullMem,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_memory_seq_add threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
