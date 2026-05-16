use std::num::NonZeroI32;
use std::os::raw::c_int;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    #[error("llama_rs_decode called with null context")]
    NullContextArg,
    #[error("llama_rs_decode called with null out_error")]
    NullOutErrorArg,
    #[error("llama_decode returned non-zero code 1: no kv cache slot was available")]
    NoKvCacheSlot,
    #[error("llama_decode returned non-zero code 2: aborted by abort callback")]
    Aborted,
    #[error("llama_decode returned non-zero code -1: n_tokens == 0")]
    NTokensZero,
    #[error("llama_decode returned unrecognized non-zero code: {code}")]
    VendoredReturnedUnrecognizedNonzeroCode { code: c_int },
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_decode threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}

impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            2 => Self::Aborted,
            -1 => Self::NTokensZero,
            error_code => Self::VendoredReturnedUnrecognizedNonzeroCode { code: error_code },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::DecodeError;

    #[test]
    fn no_kv_cache_slot_maps_from_code_one() {
        let error = DecodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, DecodeError::NoKvCacheSlot);
    }

    #[test]
    fn aborted_maps_from_code_two() {
        let error = DecodeError::from(NonZeroI32::new(2).expect("2 is non-zero"));

        assert_eq!(error, DecodeError::Aborted);
    }

    #[test]
    fn n_tokens_zero_maps_from_code_negative_one() {
        let error = DecodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, DecodeError::NTokensZero);
    }

    #[test]
    fn unrecognized_code_falls_through_to_typed_variant() {
        let error = DecodeError::from(NonZeroI32::new(42).expect("42 is non-zero"));

        assert_eq!(
            error,
            DecodeError::VendoredReturnedUnrecognizedNonzeroCode { code: 42 }
        );
    }
}
