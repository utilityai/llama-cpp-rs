//! Identity of the mmproj GGUF file the harness optionally loads for a phase.
//!
//! Same shape and semantics as [`crate::ModelSource`], but for the multimodal projection file.
//! Independent of the model's source — a test may mix any combination (HF model + local mmproj,
//! local model + HF mmproj, both local, both HF).

use std::path::PathBuf;

use anyhow::Result;

use crate::download_model::download_model;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MmprojSource {
    HuggingFace {
        repo: &'static str,
        file: &'static str,
    },
    LocalPath(&'static str),
}

impl MmprojSource {
    /// Resolves the source to an on-disk path.
    ///
    /// # Errors
    ///
    /// Returns an error if the HF download fails. `LocalPath` is infallible here — file
    /// existence is checked at load time by the mtmd context init.
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

    use super::MmprojSource;

    #[test]
    fn resolve_path_for_local_path_returns_the_literal_path() {
        let source = MmprojSource::LocalPath("/abs/mmproj.gguf");

        let resolved = source.resolve_path().expect("LocalPath resolve is infallible");

        assert_eq!(resolved, PathBuf::from("/abs/mmproj.gguf"));
    }
}
