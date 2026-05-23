use crate::llama_fixture::LlamaFixture;

/// No-op test function with the [`crate::LlamaTestFn`] signature. Always returns `Ok(())`.
///
/// Useful as a placeholder for [`crate::LlamaTestRegistration`] in unit tests that exercise
/// grouping/sorting logic without needing real trial bodies. Also covered by a self-test
/// trial so the function shows up in coverage.
///
/// # Errors
///
/// Never; always returns `Ok(())`. The `Result` return type matches `LlamaTestFn`.
pub const fn no_op(_fixture: &LlamaFixture<'_>) -> anyhow::Result<()> {
    Ok(())
}
