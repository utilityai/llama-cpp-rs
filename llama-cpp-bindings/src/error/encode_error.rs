use std::num::NonZeroI32;
use std::os::raw::c_int;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    #[error("model has no encoder")]
    ModelHasNoEncoder,
    #[error("no KV cache slot was available")]
    NoKvCacheSlot,
    #[error("encode batch is invalid (empty or initialization failure)")]
    BatchInvalid,
    #[error("encode ran out of memory")]
    EncodeOutOfMemory,
    #[error("backend compute failed during encode")]
    ComputeFailed,
    #[error("encode returned an unknown status code: {code}")]
    UnknownStatus { code: c_int },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}

impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            -1 => Self::BatchInvalid,
            error_code => Self::UnknownStatus { code: error_code },
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
    fn batch_invalid_maps_from_code_negative_one() {
        let error = EncodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, EncodeError::BatchInvalid);
    }

    #[test]
    fn unrecognized_code_falls_through_to_unknown_status() {
        let error = EncodeError::from(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(error, EncodeError::UnknownStatus { code: 99 });
    }
}
