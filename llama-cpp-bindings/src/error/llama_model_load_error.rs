use std::ffi::NulError;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum LlamaModelLoadError {
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    #[error("model file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("llama_rs_load_model_from_file called with null path")]
    NullPathArg,
    #[error("llama_rs_load_model_from_file called with null out_model")]
    NullOutModelArg,
    #[error("llama_rs_load_model_from_file called with null out_error")]
    NullOutErrorArg,
    #[error("llama_rs_load_model_from_file returned null (model failed to load)")]
    VendoredReturnedNull,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_load_model_from_file threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
