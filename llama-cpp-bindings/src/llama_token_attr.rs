use enumflags2::bitflags;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaTokenAttr {
    Unknown = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNKNOWN as _,
    Unused = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNUSED as _,
    Normal = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMAL as _,
    Control = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_CONTROL as _,
    UserDefined = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_USER_DEFINED as _,
    Byte = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_BYTE as _,
    Normalized = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMALIZED as _,
    LStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_LSTRIP as _,
    RStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_RSTRIP as _,
    SingleWord = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_SINGLE_WORD as _,
}
