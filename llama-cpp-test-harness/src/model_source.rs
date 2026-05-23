//! Identity of the GGUF file the harness loads for a phase.
//!
//! Two variants, mutually exclusive by construction:
//! - [`ModelSource::HuggingFace`] — pull via `hf-hub` (cached); the on-disk path is wherever the
//!   cache resolves to.
//! - [`ModelSource::LocalPath`] — use the file at the given absolute path verbatim; no download,
//!   no cache.
//!
//! Mutual exclusion is enforced at compile time by the enum's variant set. There is no string
//! heuristic anywhere — the proc-macro dispatches on syntactic path identifiers.

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
    /// Resolves the source to an on-disk path.
    ///
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

        let resolved = source.resolve_path().expect("LocalPath resolve is infallible");

        assert_eq!(resolved, PathBuf::from("/abs/example.gguf"));
    }
}
