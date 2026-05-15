/// A rusty wrapper around `rope_scaling_type`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    /// The scaling type is unspecified
    Unspecified = -1,
    /// No scaling
    None = 0,
    /// Linear scaling
    Linear = 1,
    /// Yarn scaling
    Yarn = 2,
}

/// Create a `RopeScalingType` from a `c_int` - returns `RopeScalingType::ScalingUnspecified` if
/// the value is not recognized.
impl From<i32> for RopeScalingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Linear,
            2 => Self::Yarn,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `RopeScalingType`.
impl From<RopeScalingType> for i32 {
    fn from(value: RopeScalingType) -> Self {
        match value {
            RopeScalingType::None => 0,
            RopeScalingType::Linear => 1,
            RopeScalingType::Yarn => 2,
            RopeScalingType::Unspecified => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RopeScalingType;

    #[test]
    fn rope_scaling_type_unknown_defaults_to_unspecified() {
        assert_eq!(RopeScalingType::from(99), RopeScalingType::Unspecified);
        assert_eq!(RopeScalingType::from(-100), RopeScalingType::Unspecified);
    }

    #[test]
    fn rope_scaling_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, RopeScalingType::Unspecified),
            (0, RopeScalingType::None),
            (1, RopeScalingType::Linear),
            (2, RopeScalingType::Yarn),
        ] {
            let from_raw = RopeScalingType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }
}
