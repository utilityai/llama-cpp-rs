use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_test_harness::LlamaFixture;

/// Tokenizes the literal "Hello world" with the fixture's model, builds a 512×1
/// batch, and decodes it into `context` to populate the KV cache.
///
/// # Errors
/// Forwards tokenization, batch construction, and [`LlamaContext::decode`] errors verbatim.
pub fn decode_hello_world(
    fixture: &LlamaFixture<'_>,
    context: &mut LlamaContext<'_>,
) -> Result<()> {
    let tokens = fixture.model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;
    Ok(())
}
