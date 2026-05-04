use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::GrammarError;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::TestFixture;
use serial_test::serial;

#[test]
#[serial]
fn dry_sampler_with_model() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let breakers: Vec<&[u8]> = vec![b"\n", b"\t"];
    let _sampler = LlamaSampler::dry(model, 1.5, 2.0, 128, 2, &breakers);

    Ok(())
}

#[test]
#[serial]
fn dry_sampler_with_null_byte_in_seq_breakers_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let breakers: Vec<&[u8]> = vec![b"hello\0world"];
    let result = LlamaSampler::dry(model, 1.5, 2.0, 128, 2, breakers);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn grammar_returns_sampler_for_valid_grammar() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = LlamaSampler::grammar(model, "root ::= \"hello\"", "root");

    assert!(sampler.is_ok());

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_returns_sampler_for_valid_grammar_with_triggers() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let trigger_words: Vec<&[u8]> = vec![b"function"];
    let sampler =
        LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", trigger_words, &[]);

    assert!(sampler.is_ok());

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_patterns_returns_sampler_for_valid_grammar_with_patterns() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let patterns = vec!["\\{.*".to_string()];
    let sampler =
        LlamaSampler::grammar_lazy_patterns(model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert!(sampler.is_ok());

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_with_root_not_found_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let trigger_words: Vec<&[u8]> = vec![b"function"];
    let result =
        LlamaSampler::grammar_lazy(model, "expr ::= \"hello\"", "root", trigger_words, &[]);

    assert!(matches!(result, Err(GrammarError::RootNotFound)));

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_with_null_byte_in_trigger_word_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let trigger_words: Vec<&[u8]> = vec![b"hel\0lo"];
    let result =
        LlamaSampler::grammar_lazy(model, "root ::= \"hello\"", "root", trigger_words, &[]);

    assert!(matches!(result, Err(GrammarError::TriggerWordNullBytes(_))));

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_patterns_with_root_not_found_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let patterns = vec!["\\{.*".to_string()];
    let result =
        LlamaSampler::grammar_lazy_patterns(model, "expr ::= \"hello\"", "root", &patterns, &[]);

    assert!(matches!(result, Err(GrammarError::RootNotFound)));

    Ok(())
}

#[test]
#[serial]
fn grammar_lazy_patterns_with_null_byte_in_pattern_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let patterns = vec!["hel\0lo".to_string()];
    let result =
        LlamaSampler::grammar_lazy_patterns(model, "root ::= \"hello\"", "root", &patterns, &[]);

    assert!(matches!(result, Err(GrammarError::GrammarNullBytes(_))));

    Ok(())
}

#[test]
#[serial]
fn llguidance_method_creates_sampler() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = LlamaSampler::llguidance(model, "regex", r"yes|no");

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn logit_bias_with_empty_biases_succeeds() {
    let result = LlamaSampler::logit_bias(0, &[]);

    assert!(result.is_ok());
}

#[test]
#[serial]
fn dry_sampler_with_root_not_found_grammar_does_not_apply() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let breakers: Vec<&[u8]> = vec![b"\n"];
    let _sampler = LlamaSampler::dry(model, 1.5, 2.0, 128, 2, &breakers);

    Ok(())
}

#[test]
#[serial]
fn sample_returns_token_after_decode() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("Hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);
    let result = sampler.sample(&context, batch.n_tokens() - 1);

    assert!(result.is_ok());

    Ok(())
}
