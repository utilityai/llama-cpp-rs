//! Utilities for working with `llama_token_type` values.

/// A rust flavored equivalent of `llama_token_type`.
#[repr(u32)]
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[allow(clippy::module_name_repetitions)]
pub enum LlamaTokenType {
    /// An undefined token type.
    Undefined = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNDEFINED,
    /// A normal token type.
    Normal = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_NORMAL,
    /// An unknown token type.
    Unknown = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNKNOWN,
    /// A control token type.
    Control = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_CONTROL,
    /// A user defined token type.
    UserDefined = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_USER_DEFINED,
    /// An unused token type.
    Unused = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNUSED,
    /// A byte token type.
    Byte = llama_cpp_sys_2::LLAMA_TOKEN_TYPE_BYTE,
}

/// A safe wrapper for converting potentially deceptive `llama_token_type` values into
/// `LlamaVocabType`.
///
/// The error branch returns the original value.
///
/// ```
/// # use std::convert::TryFrom;
/// # use std::ffi::c_uint;
/// # use std::num::TryFromIntError;
/// # use std::result::Result;
/// # use llama_cpp_2::token_type::{LlamaTokenTypeFromIntError, LlamaTokenType};
/// # fn main() -> Result<(), LlamaTokenTypeFromIntError> {
/// let llama_token_type = LlamaTokenType::try_from(0 as c_uint)?;
/// assert_eq!(llama_token_type, LlamaTokenType::Undefined);
///
/// let bad_llama_token_type = LlamaTokenType::try_from(100 as c_uint);
/// assert_eq!(Err(LlamaTokenTypeFromIntError::UnknownValue(100)), bad_llama_token_type);
/// # Ok(())
/// # }
impl TryFrom<llama_cpp_sys_2::llama_token_type> for LlamaTokenType {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_sys_2::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNDEFINED => Ok(LlamaTokenType::Undefined),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_NORMAL => Ok(LlamaTokenType::Normal),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNKNOWN => Ok(LlamaTokenType::Unknown),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_CONTROL => Ok(LlamaTokenType::Control),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_USER_DEFINED => Ok(LlamaTokenType::UserDefined),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_UNUSED => Ok(LlamaTokenType::Unused),
            llama_cpp_sys_2::LLAMA_TOKEN_TYPE_BYTE => Ok(LlamaTokenType::Byte),
            _ => Err(LlamaTokenTypeFromIntError::UnknownValue(value)),
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
