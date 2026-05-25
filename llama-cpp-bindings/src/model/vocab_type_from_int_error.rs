#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum VocabTypeFromIntError {
    #[error("Unknown Value {0}")]
    UnknownValue(llama_cpp_bindings_sys::llama_vocab_type),
}
