use std::path::Path;

use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::mtmd_context::MtmdContext;

use crate::context_params::ContextParams;

pub struct LlamaFixture<'fixture> {
    pub model: &'fixture LlamaModel,
    pub backend: &'fixture LlamaBackend,
    pub context_params: &'fixture ContextParams,
    pub mtmd_context: Option<&'fixture MtmdContext>,
    pub model_path: &'fixture Path,
}

impl LlamaFixture<'_> {
    /// # Errors
    /// Forwards [`LlamaContext::from_model`] errors verbatim.
    pub fn build_context(&self) -> Result<LlamaContext<'_>> {
        Ok(LlamaContext::from_model(
            self.model,
            self.backend,
            (*self.context_params).into_llama_context_params(),
        )?)
    }
}
