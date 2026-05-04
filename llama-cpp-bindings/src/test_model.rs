use std::env;
use std::path::PathBuf;

use anyhow::Result;

fn required_env(var_name: &str) -> Result<String> {
    env::var(var_name).map_err(|_| {
        anyhow::anyhow!(
            "Required env var {var_name} is not set. Source .env.test or set it manually."
        )
    })
}

fn hf_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_REPO")
}

fn hf_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_MODEL")
}

fn hf_mmproj() -> String {
    env::var("LLAMA_TEST_HF_MMPROJ").unwrap_or_default()
}

fn hf_embed_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_EMBED_REPO")
}

fn hf_embed_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_EMBED_MODEL")
}

fn hf_encoder_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_ENCODER_REPO")
}

fn hf_encoder_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_ENCODER_MODEL")
}

/// Downloads a file from a specific `HuggingFace` repo.
///
/// # Errors
/// Returns an error if the download fails.
pub fn download_file_from(repo: &str, filename: &str) -> Result<PathBuf> {
    download_file(repo, filename)
}

fn download_file(repo: &str, filename: &str) -> Result<PathBuf> {
    let path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(repo.to_string())
        .get(filename)?;

    Ok(path)
}

/// Downloads the configured test model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_model() -> Result<PathBuf> {
    download_file(&hf_repo()?, &hf_model()?)
}

/// Downloads the configured mmproj file from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_mmproj() -> Result<PathBuf> {
    let mmproj = hf_mmproj();

    if mmproj.is_empty() {
        anyhow::bail!("LLAMA_TEST_HF_MMPROJ is not set or empty");
    }

    download_file(&hf_repo()?, &mmproj)
}

/// Downloads the configured embedding model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_embedding_model() -> Result<PathBuf> {
    download_file(&hf_embed_repo()?, &hf_embed_model()?)
}

/// Downloads the configured encoder model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_encoder_model() -> Result<PathBuf> {
    download_file(&hf_encoder_repo()?, &hf_encoder_model()?)
}

/// Returns whether a multimodal projection model is configured.
#[must_use]
pub fn has_mmproj() -> bool {
    !hf_mmproj().is_empty()
}

/// Returns the path to the test fixtures directory.
#[must_use]
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Loads the default test model and backend.
///
/// # Errors
/// Returns an error if the backend cannot be initialized or the model cannot be loaded.
pub fn load_default_model() -> Result<(crate::llama_backend::LlamaBackend, crate::model::LlamaModel)>
{
    let backend = crate::llama_backend::LlamaBackend::init()?;
    let model_path = download_model()?;
    let model_params = crate::model::params::LlamaModelParams::default();
    let model = crate::model::LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    Ok((backend, model))
}

/// Loads the default embedding model and backend.
///
/// # Errors
/// Returns an error if the backend cannot be initialized or the embedding model cannot be loaded.
pub fn load_default_embedding_model()
-> Result<(crate::llama_backend::LlamaBackend, crate::model::LlamaModel)> {
    let backend = crate::llama_backend::LlamaBackend::init()?;
    let model_path = download_embedding_model()?;
    let model_params = crate::model::params::LlamaModelParams::default();
    let model = crate::model::LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    Ok((backend, model))
}

/// Loads the default test model, backend, and multimodal context.
///
/// # Errors
/// Returns an error if the backend cannot be initialized, the model cannot be loaded,
/// or the multimodal projection file is not configured.
pub fn load_default_mtmd() -> Result<(
    crate::llama_backend::LlamaBackend,
    crate::model::LlamaModel,
    crate::mtmd::MtmdContext,
)> {
    if !has_mmproj() {
        anyhow::bail!("MTMD tests require mmproj — set LLAMA_TEST_HF_MMPROJ");
    }

    let backend = crate::llama_backend::LlamaBackend::init()?;
    let model_path = download_model()?;
    let mmproj_path = download_mmproj()?;
    let model_params = crate::model::params::LlamaModelParams::default();
    let model = crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    let mtmd_params = crate::mtmd::MtmdContextParams::default();
    let mmproj_str = mmproj_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("mmproj path is not valid UTF-8"))?;
    let mtmd_ctx = crate::mtmd::MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)?;
    Ok((backend, model, mtmd_ctx))
}

#[cfg(test)]
mod tests {
    struct EnvVarGuard {
        name: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let original = std::env::var(name).ok();
            unsafe { std::env::set_var(name, value) };

            Self { name, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe { std::env::set_var(self.name, value) },
                None => unsafe { std::env::remove_var(self.name) },
            }
        }
    }

    #[test]
    fn required_env_returns_error_for_missing_var() {
        let result = super::required_env("LLAMA_TEST_NONEXISTENT_VAR_THAT_SHOULD_NOT_EXIST");

        assert!(result.is_err());
    }

    #[test]
    #[serial_test::serial]
    fn load_default_mtmd_fails_without_mmproj() {
        let _guard = EnvVarGuard::set("LLAMA_TEST_HF_MMPROJ", "");
        let result = super::load_default_mtmd();

        assert!(result.is_err());
    }

    #[test]
    fn download_file_with_nonexistent_file_returns_error() {
        let result =
            super::download_file("unsloth/Qwen3.5-0.8B-GGUF", "this-file-does-not-exist.gguf");

        assert!(result.is_err());
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial_test::serial]
    fn download_file_from_succeeds_for_known_repo_and_file() {
        let result =
            super::download_file_from("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf");

        assert!(result.is_ok());
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial_test::serial]
    fn download_model_returns_path_with_env_set() {
        let result = super::download_model();

        assert!(result.is_ok());
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial_test::serial]
    fn download_embedding_model_returns_path_with_env_set() {
        let result = super::download_embedding_model();

        assert!(result.is_ok());
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial_test::serial]
    fn download_encoder_model_returns_path_with_env_set() {
        let result = super::download_encoder_model();

        assert!(result.is_ok());
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial_test::serial]
    fn download_mmproj_returns_path_when_env_set() {
        let _guard = EnvVarGuard::set("LLAMA_TEST_HF_MMPROJ", "mmproj-F16.gguf");
        let result = super::download_mmproj();

        assert!(result.is_ok());
    }

    #[test]
    #[serial_test::serial]
    fn has_mmproj_reflects_env_var() {
        let _set_guard = EnvVarGuard::set("LLAMA_TEST_HF_MMPROJ", "mmproj-F16.gguf");
        assert!(super::has_mmproj());

        let _empty_guard = EnvVarGuard::set("LLAMA_TEST_HF_MMPROJ", "");
        assert!(!super::has_mmproj());
    }

    #[test]
    fn fixtures_dir_is_under_manifest() {
        let dir = super::fixtures_dir();

        assert!(dir.ends_with("fixtures"));
    }

    #[test]
    #[serial_test::serial]
    fn download_mmproj_returns_error_when_env_empty() {
        let original = std::env::var("LLAMA_TEST_HF_MMPROJ").unwrap_or_default();
        unsafe { std::env::set_var("LLAMA_TEST_HF_MMPROJ", "") };
        let result = super::download_mmproj();
        unsafe { std::env::set_var("LLAMA_TEST_HF_MMPROJ", original) };

        assert!(result.is_err());
    }
}
