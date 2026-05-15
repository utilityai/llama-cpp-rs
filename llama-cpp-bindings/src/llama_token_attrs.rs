use std::ops::{Deref, DerefMut};

use enumflags2::BitFlags;

use crate::llama_token_attr::LlamaTokenAttr;
use crate::llama_token_attrs_from_int_error::LlamaTokenAttrsFromIntError;

#[cfg(target_env = "msvc")]
const fn llama_token_type_to_u32(value: llama_cpp_bindings_sys::llama_token_type) -> u32 {
    value.cast_unsigned()
}

#[cfg(not(target_env = "msvc"))]
const fn llama_token_type_to_u32(value: llama_cpp_bindings_sys::llama_token_type) -> u32 {
    value
}

/// A set of [`LlamaTokenAttr`] flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaTokenAttrs(pub BitFlags<LlamaTokenAttr>);

impl Deref for LlamaTokenAttrs {
    type Target = BitFlags<LlamaTokenAttr>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LlamaTokenAttrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TryFrom<llama_cpp_bindings_sys::llama_token_type> for LlamaTokenAttrs {
    type Error = LlamaTokenAttrsFromIntError;

    fn try_from(value: llama_cpp_bindings_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        Ok(Self(
            BitFlags::from_bits(llama_token_type_to_u32(value)).map_err(|bit_flag_error| {
                LlamaTokenAttrsFromIntError::UnknownValue(bit_flag_error.invalid_bits())
            })?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;

    use super::{LlamaTokenAttr, LlamaTokenAttrs, LlamaTokenAttrsFromIntError};

    #[test]
    fn try_from_valid_single_attribute() {
        let attrs = LlamaTokenAttrs::try_from(llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMAL);

        assert!(attrs.is_ok());
        assert!(
            attrs
                .expect("valid attribute")
                .contains(LlamaTokenAttr::Normal)
        );
    }

    #[test]
    fn try_from_zero_produces_empty_flags() {
        let attrs = LlamaTokenAttrs::try_from(0);

        assert!(attrs.is_ok());
        assert!(attrs.expect("valid attribute").is_empty());
    }

    #[test]
    fn try_from_invalid_bits_returns_error() {
        let result = LlamaTokenAttrs::try_from(!0);

        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            LlamaTokenAttrsFromIntError::UnknownValue(_),
        ));
    }

    #[test]
    fn deref_exposes_bitflags_methods() {
        let attrs = LlamaTokenAttrs(BitFlags::from_flag(LlamaTokenAttr::Control));

        assert!(attrs.contains(LlamaTokenAttr::Control));
        assert!(!attrs.contains(LlamaTokenAttr::Normal));
    }

    #[test]
    fn deref_mut_allows_modification() {
        let mut attrs = LlamaTokenAttrs(BitFlags::empty());

        attrs.insert(LlamaTokenAttr::Byte);

        assert!(attrs.contains(LlamaTokenAttr::Byte));
    }
}
