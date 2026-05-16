#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    #[error("Wrapper received a null mtmd-context argument")]
    NullCtxArg,
    #[error("Wrapper received a null chunk argument")]
    NullChunkArg,
    #[error("mtmd_encode_chunk returned nonzero code: {code}")]
    VendoredReturnedNonzeroCode { code: i32 },
    #[error("Wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("mtmd_encode_chunk threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
