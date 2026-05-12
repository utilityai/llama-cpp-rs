use enumflags2::bitflags;

/// A rust flavored equivalent of `llama_token_type`.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaTokenAttr {
    /// Unknown token attribute.
    Unknown = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNKNOWN as _,
    /// Unused token attribute.
    Unused = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNUSED as _,
    /// Normal text token.
    Normal = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMAL as _,
    /// Control token (e.g. BOS, EOS).
    Control = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_CONTROL as _,
    /// User-defined token.
    UserDefined = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_USER_DEFINED as _,
    /// Byte-level fallback token.
    Byte = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_BYTE as _,
    /// Token with normalized text.
    Normalized = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMALIZED as _,
    /// Token with left-stripped whitespace.
    LStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_LSTRIP as _,
    /// Token with right-stripped whitespace.
    RStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_RSTRIP as _,
    /// Token representing a single word.
    SingleWord = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_SINGLE_WORD as _,
}
