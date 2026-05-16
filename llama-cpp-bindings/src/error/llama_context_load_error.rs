#[derive(Debug, thiserror::Error)]
pub enum LlamaContextLoadError {
    #[error("llama_rs_new_context_with_model called with null model")]
    NullModelArg,
    #[error("llama_rs_new_context_with_model called with null out_ctx")]
    NullOutCtxArg,
    #[error("llama_rs_new_context_with_model called with null out_error")]
    NullOutErrorArg,
    #[error("llama_rs_new_context_with_model returned null")]
    VendoredReturnedNull,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_new_context_with_model threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
