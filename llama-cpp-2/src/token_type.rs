//! Utilities for working with `llama_token_type` values.

/// A rust flavored equivalent of `llama_token_type`.
#[repr(u32)]
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[allow(clippy::module_name_repetitions, missing_docs)]
pub enum LlamaTokenAttr {
    Undefined = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNDEFINED as _,
    Unknown = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNKNOWN as _,
    Unused = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNUSED as _,
    Normal = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_NORMAL as _,
    Control = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_CONTROL as _,
    UserDefined = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_USER_DEFINED as _,
    Byte = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_BYTE as _,
    Normalized = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_NORMALIZED as _,
    LStrip = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_LSTRIP as _,
    RStrip = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_RSTRIP as _,
    SingleWord = llama_cpp_sys_2::LLAMA_TOKEN_ATTR_SINGLE_WORD as _,
}

/// A safe wrapper for converting potentially deceptive `llama_token_type` values into
/// `LlamaVocabType`.
///
/// The error branch returns the original value.
///
/// ```
/// # use std::convert::TryFrom;
/// # use std::ffi::c_int;
/// # use std::num::TryFromIntError;
/// # use std::result::Result;
/// # use llama_cpp_2::token_type::{LlamaTokenTypeFromIntError, LlamaTokenAttr};
/// # fn main() -> Result<(), LlamaTokenTypeFromIntError> {
/// let llama_token_type = LlamaTokenAttr::try_from(0 as llama_cpp_sys_2::llama_token_type)?;
/// assert_eq!(llama_token_type, LlamaTokenAttr::Undefined);
///
/// let bad_llama_token_type = LlamaTokenAttr::try_from(100 as llama_cpp_sys_2::llama_token_type);
/// assert_eq!(Err(LlamaTokenTypeFromIntError::UnknownValue(100)), bad_llama_token_type);
/// # Ok(())
/// # }
impl TryFrom<llama_cpp_sys_2::llama_token_type> for LlamaTokenAttr {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_sys_2::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNDEFINED => Ok(Self::Undefined),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNKNOWN => Ok(Self::Unknown),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_UNUSED => Ok(Self::Unused),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_NORMAL => Ok(Self::Normal),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_CONTROL => Ok(Self::Control),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_USER_DEFINED => Ok(Self::UserDefined),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_BYTE => Ok(Self::Byte),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_NORMALIZED => Ok(Self::Normalized),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_LSTRIP => Ok(Self::LStrip),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_RSTRIP => Ok(Self::RStrip),
            llama_cpp_sys_2::LLAMA_TOKEN_ATTR_SINGLE_WORD => Ok(Self::SingleWord),
            _ => Err(LlamaTokenTypeFromIntError::UnknownValue(value as _)),
        }
    }
}

/// An error type for `LlamaTokenType::try_from`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`.
    #[error("Unknown Value {0}")]
    UnknownValue(std::ffi::c_uint),
}
