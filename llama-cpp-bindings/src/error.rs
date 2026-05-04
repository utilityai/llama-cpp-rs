use std::ffi::NulError;
use std::num::NonZeroI32;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::string::FromUtf8Error;

use crate::llama_batch::BatchAddError;
use crate::mtmd::MtmdEvalError;
use crate::mtmd::mtmd_input_chunk_type::MtmdInputChunkTypeError;

/// A failable result from a llama.cpp function.
pub type Result<TValue> = std::result::Result<TValue, LlamaCppError>;

/// All errors that can occur in the llama-cpp crate.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    /// The backend was already initialized. This can generally be ignored as initializing the backend
    /// is idempotent.
    #[error("BackendAlreadyInitialized")]
    BackendAlreadyInitialized,
    /// There was an error while get the chat template from model.
    #[error("{0}")]
    ChatTemplateError(#[from] ChatTemplateError),
    /// There was an error while decoding a batch.
    #[error("{0}")]
    DecodeError(#[from] DecodeError),
    /// There was an error while encoding a batch.
    #[error("{0}")]
    EncodeError(#[from] EncodeError),
    /// There was an error loading a model.
    #[error("{0}")]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    /// There was an error creating a new model context.
    #[error("{0}")]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    /// There was an error adding a token to a batch.
    #[error["{0}"]]
    BatchAddError(#[from] BatchAddError),
    /// see [`EmbeddingsError`]
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingsError),
    /// Backend device not found
    #[error("Backend device {0} not found")]
    BackendDeviceNotFound(usize),
    /// Max devices exceeded
    #[error("Max devices exceeded. Max devices is {0}")]
    MaxDevicesExceeded(usize),
    /// Failed to convert JSON schema to grammar.
    #[error("JsonSchemaToGrammarError: {0}")]
    JsonSchemaToGrammarError(String),
    /// see [`FitError`]
    #[error(transparent)]
    FitError(#[from] FitError),
}

/// There was an error while getting the chat template from a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateError {
    /// gguf has no chat template (by that name)
    #[error("chat template not found - returned null pointer")]
    MissingTemplate,

    /// chat template contained a null byte
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// The chat template was not valid utf8.
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Failed fetching metadata value
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MetaValError {
    /// The provided string contains an unexpected null-byte
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// The returned data contains invalid UTF8 data
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),

    /// Got negative return value. This happens if the key or index queried does not exist.
    #[error("Negative return value. Likely due to a missing index or key. Got return value: {0}")]
    NegativeReturn(i32),
}

/// Failed to Load context
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    /// llama.cpp returned null
    #[error("null reference from llama.cpp")]
    NullReturn,
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// No kv cache slot was available.
    #[error("Decode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The computation was aborted by the abort callback.
    #[error("Decode Error 2: Aborted")]
    Aborted,
    /// The number of tokens in the batch was 0.
    #[error("Decode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Decode Error {0}: unknown")]
    Unknown(c_int),
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    /// No kv cache slot was available.
    #[error("Encode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Encode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Encode Error {0}: unknown")]
    Unknown(c_int),
}

/// When embedding related functions fail
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    /// Embeddings weren't enabled in the context options
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    /// Logits weren't enabled for the given token
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
    /// The given sequence index exceeds the max sequence id
    #[error("Can't use sequence embeddings with a model supporting only LLAMA_POOLING_TYPE_NONE")]
    NonePoolType,
    /// The embedding dimension does not fit into a usize.
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(#[source] std::num::TryFromIntError),
}

/// When logits-related functions fail
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LogitsError {
    /// The logits data pointer is null.
    #[error("logits data pointer is null")]
    NullLogits,
    /// The requested token index has not been initialized for logits.
    #[error("logit for token index {0} is not initialized")]
    TokenNotInitialized(i32),
    /// The token index exceeds the context size.
    #[error("token index {token_index} exceeds context size {context_size}")]
    TokenIndexExceedsContext {
        /// The token index that was requested.
        token_index: u32,
        /// The context size.
        context_size: u32,
    },
    /// The vocabulary size does not fit into a usize.
    #[error("n_vocab does not fit into usize: {0}")]
    VocabSizeOverflow(#[source] std::num::TryFromIntError),
    /// The token index does not fit into a u32.
    #[error("token_index does not fit into u32: {0}")]
    TokenIndexOverflow(#[source] std::num::TryFromIntError),
}

/// Errors that can occur when initializing a grammar sampler
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GrammarError {
    /// The grammar root was not found in the grammar string
    #[error("Grammar root not found in grammar string")]
    RootNotFound,
    /// The trigger word contains null bytes
    #[error("Trigger word contains null bytes: {0}")]
    TriggerWordNullBytes(NulError),
    /// The grammar string or root contains null bytes
    #[error("Grammar string or root contains null bytes: {0}")]
    GrammarNullBytes(NulError),
    /// A string contains null bytes
    #[error("String contains null bytes: {0}")]
    NulError(#[from] NulError),
    /// The grammar call returned null
    #[error("Grammar initialization failed: {0}")]
    NullGrammar(String),
    /// An integer value exceeded the allowed range
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
    /// An error from the llguidance library
    #[error("llguidance error: {0}")]
    LlguidanceError(String),
}

/// Errors that can occur when creating a sampling configuration.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    /// An integer value exceeded the allowed range
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
}

/// Errors that can occur when sampling a token.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SampleError {
    /// A C++ exception was thrown during sampling
    #[error("C++ exception during sampling: {0}")]
    CppException(String),

    /// An invalid argument was passed to the sampler
    #[error("Invalid argument passed to sampler")]
    InvalidArgument,
}

/// Decode a error from llama.cpp into a [`DecodeError`].
impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            2 => Self::Aborted,
            -1 => Self::NTokensZero,
            error_code => Self::Unknown(error_code),
        }
    }
}

/// Encode a error from llama.cpp into a [`EncodeError`].
impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            -1 => Self::NTokensZero,
            error_code => Self::Unknown(error_code),
        }
    }
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaModelLoadError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    /// The model file does not exist at the given path.
    #[error("model file not found: {0}")]
    FileNotFound(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterInitError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
    /// The adapter file does not exist at the given path.
    #[error("adapter file not found: {0}")]
    FileNotFound(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterSetError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// An error that can occur when converting a token to a string.
#[derive(Debug, thiserror::Error, Clone)]
#[non_exhaustive]
pub enum TokenToStringError {
    /// the token type was unknown
    #[error("Unknown Token Type")]
    UnknownTokenType,
    /// There was insufficient buffer space to convert the token to a string.
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    /// The token was not valid utf8.
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// An integer conversion failed.
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}

/// Failed to convert a string to a token sequence.
#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    /// Failed to convert a provided integer to a [`c_int`].
    CIntConversionError(#[from] std::num::TryFromIntError),
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    /// the string could not be converted to utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// An integer conversion failed.
    #[error("Integer conversion error: {0}")]
    IntConversionError(#[from] std::num::TryFromIntError),
}

/// Failed to build a [`crate::reasoning_token_classifier::ReasoningTokenClassifier`] for a model.
#[derive(Debug, thiserror::Error)]
pub enum ReasoningClassifierError {
    /// llama.cpp returned an error code from the marker detection FFI call.
    #[error("ffi error {0}")]
    FfiError(i32),
    /// The C++ side threw an exception during template analysis.
    #[error("c++ exception during template analysis: {0}")]
    AnalyzeException(String),
    /// llama.cpp returned a marker string but its bytes were not valid UTF-8.
    #[error("ffi returned non-utf8 marker bytes: {0}")]
    MarkerUtf8Error(#[from] FromUtf8Error),
    /// Tokenizing a detected marker string failed.
    #[error("marker tokenization failed: {0}")]
    MarkerTokenization(#[from] StringToTokenError),
    /// Reading token attributes for a resolved marker token failed.
    #[error("token attribute lookup failed: {0}")]
    TokenAttr(#[from] crate::token_type::LlamaTokenTypeFromIntError),
    /// The detected open-marker string did not tokenize to exactly one token.
    #[error("open marker {marker:?} tokenized to {token_count} tokens, expected 1")]
    OpenMarkerNotSingleToken {
        /// The marker string returned by llama.cpp.
        marker: String,
        /// The number of tokens the marker tokenized to.
        token_count: usize,
    },
    /// The detected close-marker string did not tokenize to exactly one token.
    #[error("close marker {marker:?} tokenized to {token_count} tokens, expected 1")]
    CloseMarkerNotSingleToken {
        /// The marker string returned by llama.cpp.
        marker: String,
        /// The number of tokens the marker tokenized to.
        token_count: usize,
    },
    /// The detected open-marker token is not registered as a special token (Control or `UserDefined`).
    #[error("open marker {marker:?} is not a registered special token")]
    OpenMarkerNotSpecial {
        /// The marker string returned by llama.cpp.
        marker: String,
    },
    /// The detected close-marker token is not registered as a special token (Control or `UserDefined`).
    #[error("close marker {marker:?} is not a registered special token")]
    CloseMarkerNotSpecial {
        /// The marker string returned by llama.cpp.
        marker: String,
    },
}

/// Failed to evaluate multimodal chunks through the request classifier.
#[derive(Debug, thiserror::Error)]
pub enum EvalMultimodalChunksError {
    /// `MtmdInputChunks::eval_chunks` returned an error.
    #[error("{0}")]
    EvalFailed(#[from] MtmdEvalError),
    /// A chunk reported a type that is not known to this binding.
    #[error("{0}")]
    UnknownChunkType(#[from] MtmdInputChunkTypeError),
    /// A chunk index that was within `chunks.len()` returned `None` from `chunks.get(index)`.
    #[error("chunk index {0} out of bounds during post-eval walk")]
    ChunkOutOfBounds(usize),
}

/// Token-usage accounting violations.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenUsageError {
    /// Cached prompt tokens cannot exceed the recorded prompt total.
    #[error(
        "cached prompt tokens would reach {cached_after} but only {prompt} prompt tokens were recorded"
    )]
    CachedExceedsPrompt {
        /// Running cached total after this would-be call.
        cached_after: u64,
        /// Currently recorded prompt-token total.
        prompt: u64,
    },
}

/// Failed to accept a token in a sampler.
#[derive(Debug, thiserror::Error)]
pub enum SamplerAcceptError {
    /// A C++ exception was thrown during accept
    #[error("C++ exception during sampler accept: {0}")]
    CppException(String),

    /// An invalid argument was passed (null sampler or null error pointer)
    #[error("Invalid argument passed to sampler accept")]
    InvalidArgument,
}

/// Errors that can occur when modifying model parameters.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelParamsError {
    /// The internal override vector has no available slot.
    #[error("No available slot in override vector")]
    NoAvailableSlot,
    /// The first override slot is not empty.
    #[error("Override slot is not empty")]
    SlotNotEmpty,
    /// A character in the key is not a valid C char.
    #[error("Invalid character in key: byte {byte}, {reason}")]
    InvalidCharacterInKey {
        /// The byte value that failed conversion.
        byte: u8,
        /// The reason the conversion failed.
        reason: String,
    },
}

/// Failed to sample a token from the data array.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenSamplingError {
    /// The sampler did not select any token.
    #[error("No token was selected by the sampler")]
    NoTokenSelected,
}

/// Returned by [`crate::model::params::LlamaModelParams::fit_params`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum FitError {
    /// Could not find allocations that fit available memory.
    #[error("could not find allocations that fit available memory")]
    Failure,
    /// A hard error occurred during fitting (e.g. model not found at the specified path,
    /// or the C++ wrapper threw an exception).
    #[error("hard error during parameter fitting")]
    Error,
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::{DecodeError, EncodeError};

    #[test]
    fn decode_error_no_kv_cache_slot() {
        let error = DecodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, DecodeError::NoKvCacheSlot);
        assert_eq!(error.to_string(), "Decode Error 1: NoKvCacheSlot");
    }

    #[test]
    fn decode_error_n_tokens_zero() {
        let error = DecodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, DecodeError::NTokensZero);
        assert_eq!(error.to_string(), "Decode Error -1: n_tokens == 0");
    }

    #[test]
    fn decode_error_aborted() {
        let error = DecodeError::from(NonZeroI32::new(2).expect("2 is non-zero"));

        assert_eq!(error, DecodeError::Aborted);
        assert_eq!(error.to_string(), "Decode Error 2: Aborted");
    }

    #[test]
    fn decode_error_unknown() {
        let error = DecodeError::from(NonZeroI32::new(42).expect("42 is non-zero"));

        assert_eq!(error, DecodeError::Unknown(42));
        assert_eq!(error.to_string(), "Decode Error 42: unknown");
    }

    #[test]
    fn encode_error_no_kv_cache_slot() {
        let error = EncodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, EncodeError::NoKvCacheSlot);
        assert_eq!(error.to_string(), "Encode Error 1: NoKvCacheSlot");
    }

    #[test]
    fn encode_error_n_tokens_zero() {
        let error = EncodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, EncodeError::NTokensZero);
        assert_eq!(error.to_string(), "Encode Error -1: n_tokens == 0");
    }

    #[test]
    fn encode_error_unknown() {
        let error = EncodeError::from(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(error, EncodeError::Unknown(99));
        assert_eq!(error.to_string(), "Encode Error 99: unknown");
    }
}
