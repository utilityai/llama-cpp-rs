use std::path::Path;

use crate::load_backends_error::LoadBackendsError;
use crate::load_backends_from_path::load_backends_from_path;

/// Compile-time path to the built GGML backend modules directory.
///
/// Populated by `llama-cpp-bindings/build.rs` from the `DEP_LLAMA_BACKENDS_DIR` cargo metadata
/// emitted by `llama-cpp-bindings-sys` when built with the `dynamic-backends` feature. `None`
/// when the metadata is missing (e.g. when this crate is built outside the cargo workspace).
pub const BACKENDS_DIR: Option<&str> = option_env!("GGML_BACKENDS_DIR");

/// Load GGML backend modules from the compile-time default directory ([`BACKENDS_DIR`]).
///
/// This is a no-op when `BACKENDS_DIR` is `None`.
///
/// # Errors
///
/// Returns [`LoadBackendsError::PathNotUtf8`] when `BACKENDS_DIR` cannot be converted to UTF-8
/// and [`LoadBackendsError::PathNullByte`] when it contains an interior null byte.
pub fn load_backends() -> Result<(), LoadBackendsError> {
    let Some(dir) = BACKENDS_DIR else {
        return Ok(());
    };

    load_backends_from_path(Path::new(dir))
}

#[cfg(test)]
mod tests {
    use super::load_backends;

    #[test]
    fn load_backends_does_not_error_with_default_backends_dir() {
        let result = load_backends();

        assert!(result.is_ok());
    }
}
