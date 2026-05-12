/// Returned by [`crate::model::vocab_type::VocabType::try_from`] when the
/// integer value does not match a known `llama_vocab_type` discriminant.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum VocabTypeFromIntError {
    /// The value is not a valid `llama_vocab_type`. Contains the int value that was invalid.
    #[error("Unknown Value {0}")]
    UnknownValue(llama_cpp_bindings_sys::llama_vocab_type),
}
