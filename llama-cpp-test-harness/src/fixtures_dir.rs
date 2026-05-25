use std::path::PathBuf;

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
