use std::path::PathBuf;

use anyhow::Result;

use crate::download_model::download_model;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelSource {
    HuggingFace {
        repo: &'static str,
        file: &'static str,
    },
    LocalPath(&'static str),
}

impl ModelSource {
    /// # Errors
    ///
    /// Returns an error if the HF download fails. `LocalPath` is infallible here — file
    /// existence is checked at load time by llama.cpp.
    pub fn resolve_path(self) -> Result<PathBuf> {
        match self {
            Self::HuggingFace { repo, file } => download_model(repo, file),
            Self::LocalPath(path) => Ok(PathBuf::from(path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::ModelSource;

    #[test]
    fn resolve_path_for_local_path_returns_the_literal_path() {
        let source = ModelSource::LocalPath("/abs/example.gguf");

        let resolved = source
            .resolve_path()
            .expect("LocalPath resolve is infallible");

        assert_eq!(resolved, PathBuf::from("/abs/example.gguf"));
    }
}
