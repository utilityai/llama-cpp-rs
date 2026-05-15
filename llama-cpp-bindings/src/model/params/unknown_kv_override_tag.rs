/// Unknown KV override tag from the FFI layer.
#[derive(Debug, thiserror::Error)]
#[error("unknown KV override tag: {0}")]
pub struct UnknownKvOverrideTag(pub llama_cpp_bindings_sys::llama_model_kv_override_type);
