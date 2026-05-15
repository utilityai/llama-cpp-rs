use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use llama_cpp_bindings::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// let converted: MtmdInputChunkType = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT.try_into().unwrap();
/// assert_eq!(text_chunk, converted);
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    /// Image input chunk
    Image = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
    /// Audio input chunk
    Audio = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_AUDIO as _,
}

impl TryFrom<llama_cpp_bindings_sys::mtmd_input_chunk_type> for MtmdInputChunkType {
    type Error = MtmdInputChunkTypeError;

    fn try_from(
        chunk_type: llama_cpp_bindings_sys::mtmd_input_chunk_type,
    ) -> Result<Self, Self::Error> {
        match chunk_type {
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT => Ok(Self::Text),
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE => Ok(Self::Image),
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_AUDIO => Ok(Self::Audio),
            unknown => Err(MtmdInputChunkTypeError(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MtmdInputChunkType;
    use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

    #[test]
    fn text_variant_converts_from_raw() {
        let result =
            MtmdInputChunkType::try_from(llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT);
        assert_eq!(result, Ok(MtmdInputChunkType::Text));
    }

    #[test]
    fn image_variant_converts_from_raw() {
        let result =
            MtmdInputChunkType::try_from(llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE);
        assert_eq!(result, Ok(MtmdInputChunkType::Image));
    }

    #[test]
    fn audio_variant_converts_from_raw() {
        let result =
            MtmdInputChunkType::try_from(llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_AUDIO);
        assert_eq!(result, Ok(MtmdInputChunkType::Audio));
    }

    #[test]
    fn unknown_value_returns_error() {
        let invalid_value = 9999;
        let result = MtmdInputChunkType::try_from(invalid_value);
        assert_eq!(result, Err(MtmdInputChunkTypeError(invalid_value)));
    }
}
