use std::num::NonZeroI32;
use std::os::raw::c_int;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    #[error("no KV cache slot was available")]
    NoKvCacheSlot,
    #[error("decode aborted by callback")]
    Aborted,
    #[error("decode batch is invalid (empty, output mismatch, or initialization failure)")]
    BatchInvalid,
    #[error("decode ran out of memory")]
    DecodeOutOfMemory,
    #[error("backend compute failed during decode")]
    ComputeFailed,
    #[error("decode returned an unknown status code: {code}")]
    UnknownStatus { code: c_int },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}

impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            2 => Self::Aborted,
            -1 => Self::BatchInvalid,
            error_code => Self::UnknownStatus { code: error_code },
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
    fn batch_invalid_maps_from_code_negative_one() {
        let error = DecodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, DecodeError::BatchInvalid);
    }

    #[test]
    fn unrecognized_code_falls_through_to_unknown_status() {
        let error = DecodeError::from(NonZeroI32::new(42).expect("42 is non-zero"));

        assert_eq!(error, DecodeError::UnknownStatus { code: 42 });
    }
}
