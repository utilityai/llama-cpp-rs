#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod batch_add_error;
pub mod chat_message_parse_outcome;
pub mod context;
pub mod error;
pub mod eval_multimodal_chunks_params;
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

pub use error::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, EmbeddingsError, EncodeError,
    EvalMultimodalChunksError, GrammarError, JsonSchemaToGrammarError, KvCacheSeqAddError,
    KvCacheSeqDivError, LlamaContextLoadError, LlamaCppError, LlamaLoraAdapterInitError,
    LlamaLoraAdapterRemoveError, LlamaLoraAdapterSetError, LlamaModelLoadError, LogitsError,
    MarkerDetectionError, MetaValError, ModelParamsError, NewLlamaChatMessageError,
    ParseChatMessageError, Result, SampleError, SamplerAcceptError, SamplingError,
    StringToTokenError, TokenSamplingError, TokenToStringError,
};

pub use chat_message_parse_outcome::ChatMessageParseOutcome;
pub use eval_multimodal_chunks_params::EvalMultimodalChunksParams;
pub use llama_backend_device::{LlamaBackendDevice, list_llama_ggml_backend_devices};
pub use llama_backend_device_type::LlamaBackendDeviceType;
pub use llama_cpp_bindings_types::{
    BracketedJsonShape, KeyValueXmlTagsShape, PairedQuoteShape, ParsedChatMessage, ParsedToolCall,
    ReasoningMarkers, TokenUsage, TokenUsageError, ToolCallArgsShape, ToolCallArguments,
    ToolCallMarkers, ToolCallValueQuote, XmlTagsShape,
};
pub use raw_chat_message::RawChatMessage;
pub use sampled_token::SampledToken;
pub use sampled_token_classifier::SampledTokenClassifier;
pub use sampled_token_section::SampledTokenSection;

pub use ffi_status_is_ok::status_is_ok;
pub use ffi_status_to_i32::status_to_i32;
pub use ggml_time_us::ggml_time_us;
pub use ingest_prompt_chunk::ingest_prompt_chunk;
pub use json_schema_to_grammar::json_schema_to_grammar;
pub use llama_time_us::llama_time_us;
pub use max_devices::max_devices;
pub use mlock_supported::mlock_supported;
pub use mmap_supported::mmap_supported;

pub use log_options::LogOptions;
pub use send_logs_to_log::send_logs_to_log;
