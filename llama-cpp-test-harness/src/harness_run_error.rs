use llama_cpp_bindings::error::LlamaCppError;
use thiserror::Error;

use crate::harness_arguments_error::HarnessArgumentsError;

#[derive(Debug, Error)]
pub enum HarnessRunError {
    #[error("failed to parse harness arguments: {0}")]
    ArgumentParsing(#[from] HarnessArgumentsError),
    #[error("failed to initialise the llama backend: {0}")]
    BackendInit(#[from] LlamaCppError),
}
