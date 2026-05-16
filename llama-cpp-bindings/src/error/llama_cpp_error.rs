use crate::batch_add_error::BatchAddError;
use crate::error::chat_template_error::ChatTemplateError;
use crate::error::decode_error::DecodeError;
use crate::error::embeddings_error::EmbeddingsError;
use crate::error::encode_error::EncodeError;
use crate::error::fit_error::FitError;
use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;
use crate::error::llama_context_load_error::LlamaContextLoadError;
use crate::error::llama_model_load_error::LlamaModelLoadError;

#[derive(Debug, thiserror::Error)]
pub enum LlamaCppError {
    #[error("BackendAlreadyInitialized")]
    BackendAlreadyInitialized,
    #[error(transparent)]
    ChatTemplateError(#[from] ChatTemplateError),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error(transparent)]
    EncodeError(#[from] EncodeError),
    #[error(transparent)]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    #[error(transparent)]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    #[error(transparent)]
    BatchAddError(#[from] BatchAddError),
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingsError),
    #[error("Backend device {0} not found")]
    BackendDeviceNotFound(usize),
    #[error("Max devices exceeded. Max devices is {0}")]
    MaxDevicesExceeded(usize),
    #[error(transparent)]
    JsonSchemaToGrammarError(#[from] JsonSchemaToGrammarError),
    #[error(transparent)]
    FitError(#[from] FitError),
}
