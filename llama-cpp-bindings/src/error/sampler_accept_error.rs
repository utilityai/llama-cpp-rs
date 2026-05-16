#[derive(Debug, thiserror::Error)]
pub enum SamplerAcceptError {
    #[error("llama_rs_sampler_accept called with null sampler")]
    NullSamplerArg,
    #[error("llama_rs_sampler_accept called with null out_error")]
    NullOutErrorArg,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_sampler_accept threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
