//! Bindings to the llama.cpp library.
//!
//! As llama.cpp is a very fast moving target, this crate does not attempt to create a stable API
//! with all the rust idioms. Instead it provided safe wrappers around nearly direct bindings to
//! llama.cpp. This makes it easier to keep up with the changes in llama.cpp, but does mean that
//! the API is not as nice as it could be.
//!
//! # Feature Flags
//!
//! - `cuda` enables CUDA gpu support.
//! - `sampler` adds the [`context::sample::sampler`] struct for a more rusty way of sampling.

pub mod context;
pub mod error;
pub mod extract_tool_call_markers_from_haystack;
pub mod ffi_error_reader;
pub mod ffi_status_is_ok;
pub mod ffi_status_to_i32;
pub mod ggml_time_us;
pub mod gguf_context;
pub mod gguf_context_error;
pub mod gguf_type;
pub mod ingest_prompt_chunk;
pub mod json_schema_to_grammar;
pub mod llama_backend;
pub mod llama_backend_device;
pub mod llama_backend_numa_strategy;
pub mod llama_batch;
pub mod llama_time_us;
pub mod llguidance_sampler;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_error;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_from_path;
pub mod log;
pub mod log_options;
pub mod max_devices;
pub mod mlock_supported;
pub mod mmap_supported;
pub mod model;
pub mod mtmd;
pub mod sampled_token;
pub mod sampled_token_classifier;
pub mod sampling;
pub mod timing;
pub mod token;
pub mod token_type;
pub mod tool_call_format;
pub mod tool_call_marker_pair;
pub mod tool_call_template_overrides;

pub use error::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, EncodeError,
    EvalMultimodalChunksError, GrammarError, LlamaContextLoadError, LlamaCppError,
    LlamaLoraAdapterInitError, LlamaLoraAdapterRemoveError, LlamaLoraAdapterSetError,
    LlamaModelLoadError, LogitsError, MarkerDetectionError, MetaValError, ModelParamsError,
    NewLlamaChatMessageError, ParseChatMessageError, Result, SampleError, SamplerAcceptError,
    SamplingError, StringToTokenError, TokenSamplingError, TokenToStringError,
};

pub use llama_backend_device::{
    LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices,
};
pub use llama_cpp_bindings_types::{
    BracketedJsonShape, PairedQuoteShape, ParsedChatMessage, ParsedToolCall, TokenUsage,
    TokenUsageError, ToolCallArgsShape, ToolCallArguments, ToolCallMarkers, ToolCallValueQuote,
    XmlTagsShape,
};
pub use sampled_token::SampledToken;
pub use sampled_token_classifier::SampledTokenClassifier;
pub use sampled_token_classifier::SampledTokenSection;

pub use ffi_status_is_ok::status_is_ok;
pub use ffi_status_to_i32::status_to_i32;
pub use ggml_time_us::ggml_time_us;
pub use ingest_prompt_chunk::ingest_prompt_chunk;
pub use json_schema_to_grammar::json_schema_to_grammar;
pub use llama_time_us::llama_time_us;
pub use max_devices::max_devices;
pub use mlock_supported::mlock_supported;
pub use mmap_supported::mmap_supported;

pub use log::send_logs_to_tracing;
pub use log_options::LogOptions;
