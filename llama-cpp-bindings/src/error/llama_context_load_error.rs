/// Failed to Load context
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    /// llama.cpp returned null
    #[error("null reference from llama.cpp")]
    NullReturn,
}
