use std::num::NonZeroI32;
use std::os::raw::c_int;

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// No kv cache slot was available.
    #[error("Decode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The computation was aborted by the abort callback.
    #[error("Decode Error 2: Aborted")]
    Aborted,
    /// The number of tokens in the batch was 0.
    #[error("Decode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Decode Error {0}: unknown")]
    Unknown(c_int),
}

/// Decode a error from llama.cpp into a [`DecodeError`].
impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            2 => Self::Aborted,
            -1 => Self::NTokensZero,
            error_code => Self::Unknown(error_code),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::DecodeError;

    #[test]
    fn decode_error_no_kv_cache_slot() {
        let error = DecodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, DecodeError::NoKvCacheSlot);
        assert_eq!(error.to_string(), "Decode Error 1: NoKvCacheSlot");
    }

    #[test]
    fn decode_error_n_tokens_zero() {
        let error = DecodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, DecodeError::NTokensZero);
        assert_eq!(error.to_string(), "Decode Error -1: n_tokens == 0");
    }

    #[test]
    fn decode_error_aborted() {
        let error = DecodeError::from(NonZeroI32::new(2).expect("2 is non-zero"));

        assert_eq!(error, DecodeError::Aborted);
        assert_eq!(error.to_string(), "Decode Error 2: Aborted");
    }

    #[test]
    fn decode_error_unknown() {
        let error = DecodeError::from(NonZeroI32::new(42).expect("42 is non-zero"));

        assert_eq!(error, DecodeError::Unknown(42));
        assert_eq!(error.to_string(), "Decode Error 42: unknown");
    }
}
