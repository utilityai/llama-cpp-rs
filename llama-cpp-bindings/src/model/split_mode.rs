use crate::model::llama_split_mode_parse_error::LlamaSplitModeParseError;

#[repr(i8)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LlamaSplitMode {
    None = LLAMA_SPLIT_MODE_NONE,
    #[default]
    Layer = LLAMA_SPLIT_MODE_LAYER,
    Row = LLAMA_SPLIT_MODE_ROW,
    Tensor = LLAMA_SPLIT_MODE_TENSOR,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the C API split mode constants are known small values that fit in i8"
)]
const LLAMA_SPLIT_MODE_NONE: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE as i8;
#[expect(
    clippy::cast_possible_truncation,
    reason = "the C API split mode constants are known small values that fit in i8"
)]
const LLAMA_SPLIT_MODE_LAYER: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER as i8;
#[expect(
    clippy::cast_possible_truncation,
    reason = "the C API split mode constants are known small values that fit in i8"
)]
const LLAMA_SPLIT_MODE_ROW: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW as i8;
#[expect(
    clippy::cast_possible_truncation,
    reason = "the C API split mode constants are known small values that fit in i8"
)]
const LLAMA_SPLIT_MODE_TENSOR: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_TENSOR as i8;

/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<i32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|convert_error| LlamaSplitModeParseError {
                value,
                context: format!("i32 to i8 conversion failed: {convert_error}"),
            })?;

        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError {
                value,
                context: format!("unknown split mode value: {value}"),
            }),
        }
    }
}

/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<u32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let clamped_value = i32::try_from(value).unwrap_or(i32::MAX);
        let i8_value = value
            .try_into()
            .map_err(|convert_error| LlamaSplitModeParseError {
                value: clamped_value,
                context: format!("u32 to i8 conversion failed: {convert_error}"),
            })?;

        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError {
                value: clamped_value,
                context: format!("unknown split mode value: {value}"),
            }),
        }
    }
}

impl From<LlamaSplitMode> for i32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE.into(),
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER.into(),
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW.into(),
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR.into(),
        }
    }
}

impl From<LlamaSplitMode> for u32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE as Self,
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER as Self,
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW as Self,
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR as Self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_NONE, LLAMA_SPLIT_MODE_ROW,
        LLAMA_SPLIT_MODE_TENSOR, LlamaSplitMode,
    };

    #[test]
    fn try_from_i32_invalid() {
        let result = LlamaSplitMode::try_from(99_i32);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.value, 99);
    }

    #[test]
    fn try_from_u32_invalid() {
        assert!(LlamaSplitMode::try_from(99_u32).is_err());
    }

    #[test]
    fn try_from_i32_none_roundtrip() {
        let mode = LlamaSplitMode::try_from(i32::from(LLAMA_SPLIT_MODE_NONE)).unwrap();

        assert_eq!(mode, LlamaSplitMode::None);
        assert_eq!(i32::from(mode), i32::from(LLAMA_SPLIT_MODE_NONE));
    }

    #[test]
    fn try_from_i32_layer_roundtrip() {
        let mode = LlamaSplitMode::try_from(i32::from(LLAMA_SPLIT_MODE_LAYER)).unwrap();

        assert_eq!(mode, LlamaSplitMode::Layer);
        assert_eq!(i32::from(mode), i32::from(LLAMA_SPLIT_MODE_LAYER));
    }

    #[test]
    fn try_from_i32_row_roundtrip() {
        let mode = LlamaSplitMode::try_from(i32::from(LLAMA_SPLIT_MODE_ROW)).unwrap();

        assert_eq!(mode, LlamaSplitMode::Row);
        assert_eq!(i32::from(mode), i32::from(LLAMA_SPLIT_MODE_ROW));
    }

    #[test]
    fn try_from_i32_tensor_roundtrip() {
        let mode = LlamaSplitMode::try_from(i32::from(LLAMA_SPLIT_MODE_TENSOR)).unwrap();

        assert_eq!(mode, LlamaSplitMode::Tensor);
        assert_eq!(i32::from(mode), i32::from(LLAMA_SPLIT_MODE_TENSOR));
    }

    #[test]
    fn try_from_u32_none_roundtrip() {
        let mode = LlamaSplitMode::try_from(LLAMA_SPLIT_MODE_NONE as u32).unwrap();

        assert_eq!(mode, LlamaSplitMode::None);
        assert_eq!(u32::from(mode), LLAMA_SPLIT_MODE_NONE as u32);
    }

    #[test]
    fn try_from_u32_layer_roundtrip() {
        let mode = LlamaSplitMode::try_from(LLAMA_SPLIT_MODE_LAYER as u32).unwrap();

        assert_eq!(mode, LlamaSplitMode::Layer);
        assert_eq!(u32::from(mode), LLAMA_SPLIT_MODE_LAYER as u32);
    }

    #[test]
    fn try_from_u32_row_roundtrip() {
        let mode = LlamaSplitMode::try_from(LLAMA_SPLIT_MODE_ROW as u32).unwrap();

        assert_eq!(mode, LlamaSplitMode::Row);
        assert_eq!(u32::from(mode), LLAMA_SPLIT_MODE_ROW as u32);
    }

    #[test]
    fn try_from_u32_tensor_roundtrip() {
        let mode = LlamaSplitMode::try_from(LLAMA_SPLIT_MODE_TENSOR as u32).unwrap();

        assert_eq!(mode, LlamaSplitMode::Tensor);
        assert_eq!(u32::from(mode), LLAMA_SPLIT_MODE_TENSOR as u32);
    }

    #[test]
    fn default_is_layer() {
        assert_eq!(LlamaSplitMode::default(), LlamaSplitMode::Layer);
    }

    #[test]
    fn try_from_i32_overflow_returns_error() {
        let result = LlamaSplitMode::try_from(i32::MAX);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .context
                .contains("i32 to i8 conversion failed")
        );
    }

    #[test]
    fn try_from_u32_overflow_returns_error() {
        let result = LlamaSplitMode::try_from(u32::MAX);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .context
                .contains("u32 to i8 conversion failed")
        );
    }
}
