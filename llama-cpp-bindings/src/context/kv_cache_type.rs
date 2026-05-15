/// A rusty wrapper around `ggml_type` for KV cache types.
#[expect(
    non_camel_case_types,
    reason = "variant names mirror llama.cpp's `enum ggml_type` symbol names verbatim so they can \
              be matched 1:1 against the C ABI without a translation table"
)]
#[expect(
    missing_docs,
    reason = "each variant denotes a quantisation flavour whose semantics are defined upstream in \
              ggml; restating the upstream spec inline would risk drifting from the source of truth"
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvCacheType {
    /// Represents an unknown or not-yet-mapped `ggml_type` and carries the raw value.
    /// When passed through FFI, the raw value is used as-is (if llama.cpp supports it,
    /// the runtime will operate with that type).
    /// This variant preserves API compatibility when new `ggml_type` values are
    /// introduced in the future.
    Unknown(llama_cpp_bindings_sys::ggml_type),
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
}

impl From<KvCacheType> for llama_cpp_bindings_sys::ggml_type {
    fn from(value: KvCacheType) -> Self {
        match value {
            KvCacheType::Unknown(raw) => raw,
            KvCacheType::F32 => llama_cpp_bindings_sys::GGML_TYPE_F32,
            KvCacheType::F16 => llama_cpp_bindings_sys::GGML_TYPE_F16,
            KvCacheType::Q4_0 => llama_cpp_bindings_sys::GGML_TYPE_Q4_0,
            KvCacheType::Q4_1 => llama_cpp_bindings_sys::GGML_TYPE_Q4_1,
            KvCacheType::Q5_0 => llama_cpp_bindings_sys::GGML_TYPE_Q5_0,
            KvCacheType::Q5_1 => llama_cpp_bindings_sys::GGML_TYPE_Q5_1,
            KvCacheType::Q8_0 => llama_cpp_bindings_sys::GGML_TYPE_Q8_0,
            KvCacheType::Q8_1 => llama_cpp_bindings_sys::GGML_TYPE_Q8_1,
            KvCacheType::Q2_K => llama_cpp_bindings_sys::GGML_TYPE_Q2_K,
            KvCacheType::Q3_K => llama_cpp_bindings_sys::GGML_TYPE_Q3_K,
            KvCacheType::Q4_K => llama_cpp_bindings_sys::GGML_TYPE_Q4_K,
            KvCacheType::Q5_K => llama_cpp_bindings_sys::GGML_TYPE_Q5_K,
            KvCacheType::Q6_K => llama_cpp_bindings_sys::GGML_TYPE_Q6_K,
            KvCacheType::Q8_K => llama_cpp_bindings_sys::GGML_TYPE_Q8_K,
            KvCacheType::IQ2_XXS => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS,
            KvCacheType::IQ2_XS => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS,
            KvCacheType::IQ3_XXS => llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS,
            KvCacheType::IQ1_S => llama_cpp_bindings_sys::GGML_TYPE_IQ1_S,
            KvCacheType::IQ4_NL => llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL,
            KvCacheType::IQ3_S => llama_cpp_bindings_sys::GGML_TYPE_IQ3_S,
            KvCacheType::IQ2_S => llama_cpp_bindings_sys::GGML_TYPE_IQ2_S,
            KvCacheType::IQ4_XS => llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS,
            KvCacheType::I8 => llama_cpp_bindings_sys::GGML_TYPE_I8,
            KvCacheType::I16 => llama_cpp_bindings_sys::GGML_TYPE_I16,
            KvCacheType::I32 => llama_cpp_bindings_sys::GGML_TYPE_I32,
            KvCacheType::I64 => llama_cpp_bindings_sys::GGML_TYPE_I64,
            KvCacheType::F64 => llama_cpp_bindings_sys::GGML_TYPE_F64,
            KvCacheType::IQ1_M => llama_cpp_bindings_sys::GGML_TYPE_IQ1_M,
            KvCacheType::BF16 => llama_cpp_bindings_sys::GGML_TYPE_BF16,
            KvCacheType::TQ1_0 => llama_cpp_bindings_sys::GGML_TYPE_TQ1_0,
            KvCacheType::TQ2_0 => llama_cpp_bindings_sys::GGML_TYPE_TQ2_0,
            KvCacheType::MXFP4 => llama_cpp_bindings_sys::GGML_TYPE_MXFP4,
        }
    }
}

impl From<llama_cpp_bindings_sys::ggml_type> for KvCacheType {
    fn from(value: llama_cpp_bindings_sys::ggml_type) -> Self {
        match value {
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F32 => Self::F32,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F16 => Self::F16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_0 => Self::Q4_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_1 => Self::Q4_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_0 => Self::Q5_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_1 => Self::Q5_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_0 => Self::Q8_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_1 => Self::Q8_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q2_K => Self::Q2_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q3_K => Self::Q3_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_K => Self::Q4_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_K => Self::Q5_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q6_K => Self::Q6_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_K => Self::Q8_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS => Self::IQ2_XXS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS => Self::IQ2_XS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS => Self::IQ3_XXS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ1_S => Self::IQ1_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL => Self::IQ4_NL,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ3_S => Self::IQ3_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_S => Self::IQ2_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS => Self::IQ4_XS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I8 => Self::I8,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I16 => Self::I16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I32 => Self::I32,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I64 => Self::I64,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F64 => Self::F64,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ1_M => Self::IQ1_M,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_BF16 => Self::BF16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_TQ1_0 => Self::TQ1_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_TQ2_0 => Self::TQ2_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_MXFP4 => Self::MXFP4,
            _ => Self::Unknown(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheType;

    #[test]
    fn kv_cache_type_unknown_preserves_raw_value() {
        let unknown_raw: llama_cpp_bindings_sys::ggml_type = 99999;
        let cache_type = KvCacheType::from(unknown_raw);

        assert_eq!(cache_type, KvCacheType::Unknown(99999));

        let back: llama_cpp_bindings_sys::ggml_type = cache_type.into();

        assert_eq!(back, 99999);
    }

    #[test]
    fn kv_cache_type_all_known_variants_roundtrip() {
        let all_variants = [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::Q4_0,
            KvCacheType::Q4_1,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
            KvCacheType::Q8_0,
            KvCacheType::Q8_1,
            KvCacheType::Q2_K,
            KvCacheType::Q3_K,
            KvCacheType::Q4_K,
            KvCacheType::Q5_K,
            KvCacheType::Q6_K,
            KvCacheType::Q8_K,
            KvCacheType::IQ2_XXS,
            KvCacheType::IQ2_XS,
            KvCacheType::IQ3_XXS,
            KvCacheType::IQ1_S,
            KvCacheType::IQ4_NL,
            KvCacheType::IQ3_S,
            KvCacheType::IQ2_S,
            KvCacheType::IQ4_XS,
            KvCacheType::I8,
            KvCacheType::I16,
            KvCacheType::I32,
            KvCacheType::I64,
            KvCacheType::F64,
            KvCacheType::IQ1_M,
            KvCacheType::BF16,
            KvCacheType::TQ1_0,
            KvCacheType::TQ2_0,
            KvCacheType::MXFP4,
        ];

        for variant in all_variants {
            let ggml_type: llama_cpp_bindings_sys::ggml_type = variant.into();
            let back = KvCacheType::from(ggml_type);

            assert_eq!(back, variant);
        }
    }
}
