/// Returned by [`crate::llama_token_attrs::LlamaTokenAttrs::try_from`] when the
/// integer bit pattern contains bits not defined by
/// [`crate::llama_token_attr::LlamaTokenAttr`].
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenAttrsFromIntError {
    /// The value is not a valid `llama_token_type`.
    #[error("Unknown Value {0}")]
    UnknownValue(std::ffi::c_uint),
}
