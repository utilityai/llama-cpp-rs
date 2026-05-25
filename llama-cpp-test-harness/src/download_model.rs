use std::path::PathBuf;

use anyhow::Result;

/// Downloads a single file from a Hugging Face repo via `hf-hub`'s sync API.
///
/// # Errors
///
/// Returns an error if the HF client cannot be built or the file cannot be downloaded
/// (e.g., the repo or file does not exist, or network access fails). `hf-hub`'s error type
/// already carries the repo and file in its messages, so no extra context is added here.
pub fn download_model(repo: &str, file: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()?;
    Ok(api.model(repo.to_owned()).get(file)?)
}

#[cfg(test)]
mod tests {
    use super::download_model;

    #[test]
    fn missing_file_in_real_repo_returns_error() {
        let result = download_model("unsloth/Qwen3.5-0.8B-GGUF", "this-file-does-not-exist.gguf");

        assert!(result.is_err());
    }
}
