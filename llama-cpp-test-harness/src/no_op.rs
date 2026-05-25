use crate::llama_fixture::LlamaFixture;

/// # Errors
///
/// Never; always returns `Ok(())`. The `Result` return type matches `LlamaTestFn`.
pub const fn no_op(_fixture: &LlamaFixture<'_>) -> anyhow::Result<()> {
    Ok(())
}
