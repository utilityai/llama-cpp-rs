use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdContext;

pub struct PhaseState {
    pub mtmd_context: Option<MtmdContext>,
    pub model: LlamaModel,
    pub backend: Arc<LlamaBackend>,
    pub model_path: PathBuf,
}
