//! Process-wide cached fixture for LLM-backed integration tests.

use std::sync::OnceLock;

use anyhow::Result;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;

use crate::gpu_backend::inference_model_params;
use crate::gpu_backend::require_compiled_backends_present;
use crate::test_model;

/// Shared test resources reused across LLM-backed integration tests in a single process.
///
/// The backend and the default model load eagerly on first access; the embedding model and
/// multimodal context load lazily, only when a test asks for them. The fixture lives for the
/// duration of the test process so the GGUF files are mapped into memory exactly once.
pub struct TestFixture {
    backend: LlamaBackend,
    default_model: LlamaModel,
    embedding_model: OnceLock<LlamaModel>,
    mtmd_context: OnceLock<MtmdContext>,
}

impl TestFixture {
    /// Returns the process-wide fixture, loading on first call.
    ///
    /// # Panics
    /// Panics if the backend or default model cannot be loaded — that is an
    /// unrecoverable test-setup failure and there is no meaningful continuation.
    #[must_use]
    pub fn shared() -> &'static Self {
        static FIXTURE: OnceLock<TestFixture> = OnceLock::new();

        FIXTURE.get_or_init(|| Self::load().expect("test fixture: load failed"))
    }

    fn load() -> Result<Self> {
        let backend = LlamaBackend::init()?;
        require_compiled_backends_present()?;
        let default_model = Self::load_default_model(&backend)?;

        Ok(Self {
            backend,
            default_model,
            embedding_model: OnceLock::new(),
            mtmd_context: OnceLock::new(),
        })
    }

    fn load_default_model(backend: &LlamaBackend) -> Result<LlamaModel> {
        let path = test_model::download_model()?;
        let params = inference_model_params();

        Ok(LlamaModel::load_from_file(backend, &path, &params)?)
    }

    /// Returns the backend shared by every cached resource on this fixture.
    #[must_use]
    pub const fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    /// Returns the default test model.
    #[must_use]
    pub const fn default_model(&self) -> &LlamaModel {
        &self.default_model
    }

    /// Returns the embedding model, loading it on first call.
    ///
    /// # Errors
    /// Returns an error if the required environment variables are not set or the model
    /// cannot be downloaded or loaded.
    ///
    /// # Panics
    /// Panics only if the just-stored value cannot be read back (impossible in practice).
    pub fn embedding_model(&self) -> Result<&LlamaModel> {
        if let Some(model) = self.embedding_model.get() {
            return Ok(model);
        }

        let model = self.load_embedding_model()?;
        let _ = self.embedding_model.set(model);

        Ok(self
            .embedding_model
            .get()
            .expect("test fixture: embedding model just set"))
    }

    fn load_embedding_model(&self) -> Result<LlamaModel> {
        let path = test_model::download_embedding_model()?;
        let params = LlamaModelParams::default();

        Ok(LlamaModel::load_from_file(&self.backend, &path, &params)?)
    }

    /// Returns the multimodal context, loading it on first call.
    ///
    /// # Errors
    /// Returns an error if `LLAMA_TEST_HF_MMPROJ` is unset or the context cannot be initialized.
    ///
    /// # Panics
    /// Panics only if the just-stored value cannot be read back (impossible in practice).
    pub fn mtmd_context(&self) -> Result<&MtmdContext> {
        if !test_model::has_mmproj() {
            anyhow::bail!("mtmd tests require LLAMA_TEST_HF_MMPROJ to be set");
        }
        if let Some(ctx) = self.mtmd_context.get() {
            return Ok(ctx);
        }

        let ctx = self.load_mtmd_context()?;
        let _ = self.mtmd_context.set(ctx);

        Ok(self
            .mtmd_context
            .get()
            .expect("test fixture: mtmd context just set"))
    }

    fn load_mtmd_context(&self) -> Result<MtmdContext> {
        let mmproj_path = test_model::download_mmproj()?;
        let mmproj_str = mmproj_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("mmproj path is not valid UTF-8"))?;
        let params = MtmdContextParams::default();

        Ok(MtmdContext::init_from_file(
            mmproj_str,
            &self.default_model,
            &params,
        )?)
    }
}
