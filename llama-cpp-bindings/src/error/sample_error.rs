#[derive(Debug, thiserror::Error)]
pub enum SampleError {
    #[error("llama_rs_sampler_sample called with null sampler")]
    NullSamplerArg,
    #[error("llama_rs_sampler_sample called with null context")]
    NullCtxArg,
    #[error("llama_rs_sampler_sample called with null out_token")]
    NullOutTokenArg,
    #[error("llama_rs_sampler_sample called with null out_error")]
    NullOutErrorArg,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_sampler_sample threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
