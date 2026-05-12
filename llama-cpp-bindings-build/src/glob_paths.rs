use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GlobPathsError {
    #[error("invalid glob pattern {pattern:?}: {source}")]
    InvalidPattern {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },
    #[error("glob entry failed for pattern {pattern:?}: {source}")]
    EntryError {
        pattern: String,
        #[source]
        source: glob::GlobError,
    },
    #[error("no files matched glob pattern {pattern:?}")]
    NoMatches { pattern: String },
}

pub fn collect_paths(pattern: &str) -> Result<Vec<PathBuf>, GlobPathsError> {
    let entries = glob::glob(pattern).map_err(|source| GlobPathsError::InvalidPattern {
        pattern: pattern.to_string(),
        source,
    })?;

    let mut paths = Vec::new();

    for entry in entries {
        let path = entry.map_err(|source| GlobPathsError::EntryError {
            pattern: pattern.to_string(),
            source,
        })?;

        paths.push(path);
    }

    if paths.is_empty() {
        return Err(GlobPathsError::NoMatches {
            pattern: pattern.to_string(),
        });
    }

    paths.sort();

    Ok(paths)
}
