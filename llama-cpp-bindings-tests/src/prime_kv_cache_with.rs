use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// # Errors
/// Forwards tokenization, batch construction, and [`LlamaContext::decode`] errors verbatim.
pub fn prime_kv_cache_with(
    fixture: &LlamaFixture<'_>,
    context: &mut LlamaContext<'_>,
    text: &str,
    batch_capacity: usize,
) -> Result<()> {
    let tokens = fixture.model.str_to_token(text, AddBos::Always)?;
    let mut batch = LlamaBatch::new(batch_capacity, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;
    Ok(())
}
