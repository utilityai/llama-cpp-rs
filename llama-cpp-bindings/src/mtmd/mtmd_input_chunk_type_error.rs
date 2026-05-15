/// Error when converting from an unknown MTMD input chunk type value.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("Unknown MTMD input chunk type: {0}")]
pub struct MtmdInputChunkTypeError(pub llama_cpp_bindings_sys::mtmd_input_chunk_type);
