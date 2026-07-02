#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]

pub mod batch_add_error;
pub mod chat_message_parse_outcome;
pub mod context;
pub mod error;
pub mod eval_multimodal_chunks_params;
pub mod extract_reasoning_markers_from_probe_renders;
pub mod extract_tool_call_markers_from_haystack;
pub mod ffi_error_reader;
pub mod ffi_status_is_ok;
pub mod ffi_status_to_i32;
pub mod ggml_time_us;
pub mod gguf_context;
pub mod gguf_context_error;
pub mod gguf_type;
pub mod grammar_matcher;
pub mod ingest_outcome;
pub mod ingest_prompt_chunk;
pub mod invalid_numa_strategy;
pub mod json_schema_to_grammar;
pub mod llama_backend;
pub mod llama_backend_device;
pub mod llama_backend_device_type;
pub mod llama_backend_numa_strategy;
pub mod llama_batch;
pub mod llama_time_us;
pub mod llama_token_attr;
pub mod llama_token_attrs;
pub mod llama_token_attrs_from_int_error;
pub mod llguidance_sampler;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_error;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_from_path;
pub mod log_options;
pub mod marker_kind;
pub mod mask_outcome;
pub mod max_devices;
pub mod mlock_supported;
pub mod mmap_supported;
pub mod model;
pub mod mtmd;
pub mod raw_chat_message;
pub mod resolved_tool_call_markers;
pub mod sampled_token;
pub mod sampled_token_classifier;
pub mod sampled_token_section;
pub mod sampling;
pub mod send_logs_to_log;
pub mod streaming_json_probe;
pub mod streaming_markers;
pub mod timing;
pub mod token;
pub mod tool_call_format;
pub mod tool_call_marker_pair;
pub mod tool_call_template_overrides;
