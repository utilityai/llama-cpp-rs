use crate::batch_add_error::BatchAddError;
use crate::error::chat_template_error::ChatTemplateError;
use crate::error::decode_error::DecodeError;
use crate::error::embeddings_error::EmbeddingsError;
use crate::error::encode_error::EncodeError;
use crate::error::fit_error::FitError;
use crate::error::llama_context_load_error::LlamaContextLoadError;
use crate::error::llama_model_load_error::LlamaModelLoadError;

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
