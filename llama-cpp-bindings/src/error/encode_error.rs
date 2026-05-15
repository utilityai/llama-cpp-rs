use std::num::NonZeroI32;
use std::os::raw::c_int;

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    /// No kv cache slot was available.
    #[error("Encode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Encode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Encode Error {0}: unknown")]
    Unknown(c_int),
}

/// Encode a error from llama.cpp into a [`EncodeError`].
impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            -1 => Self::NTokensZero,
            error_code => Self::Unknown(error_code),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::EncodeError;

    #[test]
    fn encode_error_no_kv_cache_slot() {
        let error = EncodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, EncodeError::NoKvCacheSlot);
        assert_eq!(error.to_string(), "Encode Error 1: NoKvCacheSlot");
    }

    #[test]
    fn encode_error_n_tokens_zero() {
        let error = EncodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, EncodeError::NTokensZero);
        assert_eq!(error.to_string(), "Encode Error -1: n_tokens == 0");
    }

    #[test]
    fn encode_error_unknown() {
        let error = EncodeError::from(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(error, EncodeError::Unknown(99));
        assert_eq!(error.to_string(), "Encode Error 99: unknown");
    }
}
