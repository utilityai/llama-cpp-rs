#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}
