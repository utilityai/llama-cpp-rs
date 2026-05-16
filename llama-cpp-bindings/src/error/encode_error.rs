use std::num::NonZeroI32;
use std::os::raw::c_int;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    #[error("llama_rs_encode called with null context")]
    NullContextArg,
    #[error("llama_rs_encode invoked on a model that has no encoder")]
    ModelHasNoEncoder,
    #[error("llama_encode returned non-zero code 1: no kv cache slot was available")]
    NoKvCacheSlot,
    #[error("llama_encode returned non-zero code -1: n_tokens == 0")]
    NTokensZero,
    #[error("llama_encode returned unrecognized non-zero code: {code}")]
    VendoredReturnedUnrecognizedNonzeroCode { code: c_int },
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_encode threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}

impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            -1 => Self::NTokensZero,
            error_code => Self::VendoredReturnedUnrecognizedNonzeroCode { code: error_code },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::EncodeError;

    #[test]
    fn no_kv_cache_slot_maps_from_code_one() {
        let error = EncodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, EncodeError::NoKvCacheSlot);
    }

    #[test]
    fn n_tokens_zero_maps_from_code_negative_one() {
        let error = EncodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, EncodeError::NTokensZero);
    }

    #[test]
    fn unrecognized_code_falls_through_to_typed_variant() {
        let error = EncodeError::from(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(
            error,
            EncodeError::VendoredReturnedUnrecognizedNonzeroCode { code: 99 }
        );
    }
}
