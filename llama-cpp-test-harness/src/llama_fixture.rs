use std::path::Path;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdContext;

use crate::context_params::ContextParams;

pub struct LlamaFixture<'fixture> {
    pub model: &'fixture LlamaModel,
    pub backend: &'fixture LlamaBackend,
    pub context_params: &'fixture ContextParams,
    pub mtmd_context: Option<&'fixture MtmdContext>,
    pub model_path: &'fixture Path,
}
