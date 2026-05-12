use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::Weak;

use anyhow::Result;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;

use crate::gpu_backend::inference_model_params;
use crate::gpu_backend::require_compiled_backends_present;
use crate::test_model;

static SHARED: Mutex<Weak<FixtureSessionInner>> = Mutex::new(Weak::new());

struct FixtureSessionInner {
    mtmd_context: OnceLock<MtmdContext>,
    embedding_model: OnceLock<LlamaModel>,
    default_model: LlamaModel,
    backend: LlamaBackend,
}

impl FixtureSessionInner {
    fn load() -> Result<Self> {
        let backend = LlamaBackend::init()?;
        require_compiled_backends_present()?;
        let default_model = Self::load_default_model(&backend)?;

        Ok(Self {
            mtmd_context: OnceLock::new(),
            embedding_model: OnceLock::new(),
            default_model,
            backend,
        })
    }

    fn load_default_model(backend: &LlamaBackend) -> Result<LlamaModel> {
        let path = test_model::download_model()?;
        let params = inference_model_params();

        Ok(LlamaModel::load_from_file(backend, &path, &params)?)
    }

    fn load_embedding_model(&self) -> Result<LlamaModel> {
        let path = test_model::download_embedding_model()?;
        let params = LlamaModelParams::default();

        Ok(LlamaModel::load_from_file(&self.backend, &path, &params)?)
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

pub struct FixtureSession {
    inner: Arc<FixtureSessionInner>,
}

impl FixtureSession {
    /// Opens a session against the shared fixture, loading on first call or
    /// after the previous session has been fully dropped.
    ///
    /// # Errors
    /// Returns an error if the backend or default model cannot be loaded.
    ///
    /// # Panics
    /// Panics if the shared mutex is poisoned by a prior load failure.
    pub fn open() -> Result<Self> {
        let inner = {
            let mut shared = SHARED.lock().expect("fixture singleton mutex poisoned");
            if let Some(existing) = shared.upgrade() {
                existing
            } else {
                let new_inner = Arc::new(FixtureSessionInner::load()?);
                *shared = Arc::downgrade(&new_inner);
                new_inner
            }
        };

        Ok(Self { inner })
    }

    #[must_use]
    pub fn backend(&self) -> &LlamaBackend {
        &self.inner.backend
    }

    #[must_use]
    pub fn default_model(&self) -> &LlamaModel {
        &self.inner.default_model
    }

    /// Returns the embedding model, loading it on first call.
    ///
    /// # Errors
    /// Returns an error if the required environment variables are not set or the
    /// model cannot be downloaded or loaded.
    ///
    /// # Panics
    /// Panics only if the just-stored value cannot be read back, which cannot
    /// happen in practice.
    pub fn embedding_model(&self) -> Result<&LlamaModel> {
        if let Some(model) = self.inner.embedding_model.get() {
            return Ok(model);
        }

        let model = self.inner.load_embedding_model()?;
        let _ = self.inner.embedding_model.set(model);

        Ok(self
            .inner
            .embedding_model
            .get()
            .expect("embedding model just set"))
    }

    /// Returns the multimodal context, loading it on first call.
    ///
    /// # Errors
    /// Returns an error if `LLAMA_TEST_HF_MMPROJ` is unset or the context cannot
    /// be initialized.
    ///
    /// # Panics
    /// Panics only if the just-stored value cannot be read back, which cannot
    /// happen in practice.
    pub fn mtmd_context(&self) -> Result<&MtmdContext> {
        if !test_model::has_mmproj() {
            anyhow::bail!("mtmd tests require LLAMA_TEST_HF_MMPROJ to be set");
        }
        if let Some(ctx) = self.inner.mtmd_context.get() {
            return Ok(ctx);
        }

        let ctx = self.inner.load_mtmd_context()?;
        let _ = self.inner.mtmd_context.set(ctx);

        Ok(self
            .inner
            .mtmd_context
            .get()
            .expect("mtmd context just set"))
    }
}
