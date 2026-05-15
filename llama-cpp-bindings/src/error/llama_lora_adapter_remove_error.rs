/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}
