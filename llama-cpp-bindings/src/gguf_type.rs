//! GGUF value types.

/// The type of a value stored in a GGUF key-value pair.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgufType {
    /// 8-bit unsigned integer
    Uint8 = 0,
    /// 8-bit signed integer
    Int8 = 1,
    /// 16-bit unsigned integer
    Uint16 = 2,
    /// 16-bit signed integer
    Int16 = 3,
    /// 32-bit unsigned integer
    Uint32 = 4,
    /// 32-bit signed integer
    Int32 = 5,
    /// 32-bit floating point
    Float32 = 6,
    /// Boolean
    Bool = 7,
    /// String
    String = 8,
    /// Array
    Array = 9,
    /// 64-bit unsigned integer
    Uint64 = 10,
    /// 64-bit signed integer
    Int64 = 11,
    /// 64-bit floating point
    Float64 = 12,
}

impl GgufType {
    /// Converts from the raw `gguf_type` value. Returns None for unknown types.
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

    /// Converts to the raw `gguf_type` value.
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
