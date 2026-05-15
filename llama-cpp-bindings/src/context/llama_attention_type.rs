/// A rusty wrapper around `LLAMA_ATTENTION_TYPE`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaAttentionType {
    /// The attention type is unspecified
    Unspecified = -1,
    /// Causal attention
    Causal = 0,
    /// Non-causal attention
    NonCausal = 1,
}

impl From<i32> for LlamaAttentionType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Causal,
            1 => Self::NonCausal,
            _ => Self::Unspecified,
        }
    }
}

impl From<LlamaAttentionType> for i32 {
    fn from(value: LlamaAttentionType) -> Self {
        match value {
            LlamaAttentionType::Causal => 0,
            LlamaAttentionType::NonCausal => 1,
            LlamaAttentionType::Unspecified => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaAttentionType;

    #[test]
    fn attention_type_unknown_defaults_to_unspecified() {
        assert_eq!(
            LlamaAttentionType::from(99),
            LlamaAttentionType::Unspecified
        );
        assert_eq!(
            LlamaAttentionType::from(-50),
            LlamaAttentionType::Unspecified
        );
    }

    #[test]
    fn attention_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, LlamaAttentionType::Unspecified),
            (0, LlamaAttentionType::Causal),
            (1, LlamaAttentionType::NonCausal),
        ] {
            let from_raw = LlamaAttentionType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }
}
