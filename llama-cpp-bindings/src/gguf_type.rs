#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    #[must_use]
    pub const fn from_raw(value: llama_cpp_bindings_sys::gguf_type) -> Option<Self> {
        match value {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_raw(self) -> llama_cpp_bindings_sys::gguf_type {
        self as llama_cpp_bindings_sys::gguf_type
    }
}

#[cfg(test)]
mod tests {
    use super::GgufType;

    #[test]
    fn from_raw_maps_all_known_types() {
        assert_eq!(GgufType::from_raw(0), Some(GgufType::Uint8));
        assert_eq!(GgufType::from_raw(1), Some(GgufType::Int8));
        assert_eq!(GgufType::from_raw(2), Some(GgufType::Uint16));
        assert_eq!(GgufType::from_raw(3), Some(GgufType::Int16));
        assert_eq!(GgufType::from_raw(4), Some(GgufType::Uint32));
        assert_eq!(GgufType::from_raw(5), Some(GgufType::Int32));
        assert_eq!(GgufType::from_raw(6), Some(GgufType::Float32));
        assert_eq!(GgufType::from_raw(7), Some(GgufType::Bool));
        assert_eq!(GgufType::from_raw(8), Some(GgufType::String));
        assert_eq!(GgufType::from_raw(9), Some(GgufType::Array));
        assert_eq!(GgufType::from_raw(10), Some(GgufType::Uint64));
        assert_eq!(GgufType::from_raw(11), Some(GgufType::Int64));
        assert_eq!(GgufType::from_raw(12), Some(GgufType::Float64));
    }

    #[test]
    fn from_raw_returns_none_for_unknown() {
        assert_eq!(GgufType::from_raw(99), None);
        assert_eq!(
            GgufType::from_raw(llama_cpp_bindings_sys::gguf_type::MAX),
            None,
        );
    }

    #[test]
    fn to_raw_roundtrips() {
        for raw in 0..=12 {
            let gguf_type = GgufType::from_raw(raw).unwrap();
            assert_eq!(gguf_type.to_raw(), raw);
        }
    }
}
