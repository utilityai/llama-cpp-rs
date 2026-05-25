#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterSetError {
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}
