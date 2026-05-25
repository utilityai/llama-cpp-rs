use crate::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    Text = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    Image = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
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
