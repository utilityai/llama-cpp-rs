#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

impl From<i32> for LlamaPoolingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Mean,
            2 => Self::Cls,
            3 => Self::Last,
            4 => Self::Rank,
            _ => Self::Unspecified,
        }
    }
}

impl From<LlamaPoolingType> for i32 {
    fn from(value: LlamaPoolingType) -> Self {
        match value {
            LlamaPoolingType::None => 0,
            LlamaPoolingType::Mean => 1,
            LlamaPoolingType::Cls => 2,
            LlamaPoolingType::Last => 3,
            LlamaPoolingType::Rank => 4,
            LlamaPoolingType::Unspecified => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaPoolingType;

    #[test]
    fn pooling_type_unknown_defaults_to_unspecified() {
        assert_eq!(LlamaPoolingType::from(99), LlamaPoolingType::Unspecified);
        assert_eq!(LlamaPoolingType::from(-50), LlamaPoolingType::Unspecified);
    }

    #[test]
    fn pooling_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, LlamaPoolingType::Unspecified),
            (0, LlamaPoolingType::None),
            (1, LlamaPoolingType::Mean),
            (2, LlamaPoolingType::Cls),
            (3, LlamaPoolingType::Last),
            (4, LlamaPoolingType::Rank),
        ] {
            let from_raw = LlamaPoolingType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }
}
