//! Path helper for image and audio fixtures used by multimodal integration tests.

use std::path::PathBuf;

/// Returns the absolute path to the test fixtures directory.
#[must_use]
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixtures_dir_is_under_manifest() {
        let dir = super::fixtures_dir();
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        assert!(dir.starts_with(manifest));
    }
}
