use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::prime_kv_cache_with::prime_kv_cache_with;

/// # Errors
/// Forwards tokenization, batch construction, and [`LlamaContext::decode`] errors verbatim.
pub fn prime_kv_cache(fixture: &LlamaFixture<'_>, context: &mut LlamaContext<'_>) -> Result<()> {
    prime_kv_cache_with(fixture, context, "Hello world", 512)
}
