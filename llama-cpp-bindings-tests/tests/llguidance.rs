use std::ffi::CStr;
use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::llguidance_sampler::create_llg_sampler;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::TestFixture;
use serial_test::serial;

const JSON_SCHEMA: &str =
    r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
const REGEX_GRAMMAR: &str = r"yes|no";
const LARK_GRAMMAR: &str = r#"start: "yes" | "no""#;

#[test]
#[serial]
fn creates_sampler_with_valid_json_schema() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = create_llg_sampler(model, "json", JSON_SCHEMA)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[test]
#[serial]
fn creates_sampler_with_valid_regex_grammar() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[test]
#[serial]
fn creates_sampler_with_valid_lark_grammar() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = create_llg_sampler(model, "lark", LARK_GRAMMAR)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[test]
#[serial]
fn returns_error_for_unknown_grammar_kind() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = create_llg_sampler(model, "not_a_real_kind", "anything");

    assert!(result.is_err());
}

#[test]
#[serial]
fn returns_error_for_malformed_json_schema() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = create_llg_sampler(model, "json", "{this is not valid json");

    assert!(result.is_err());
}

#[test]
#[serial]
fn returns_error_for_malformed_regex() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = create_llg_sampler(model, "regex", "[invalid");

    assert!(result.is_err());
}

#[test]
#[serial]
fn name_callback_returns_llguidance() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    let name_ptr = unsafe { llama_cpp_bindings_sys::llama_sampler_name(sampler.sampler) };
    assert!(!name_ptr.is_null());
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?;

    assert_eq!(name, "llguidance");

    Ok(())
}

#[test]
#[serial]
fn reset_clears_sampler_state() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mut sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    sampler.reset();

    Ok(())
}

#[test]
#[serial]
fn clone_via_ffi_creates_independent_sampler() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    let cloned = unsafe { llama_cpp_bindings_sys::llama_sampler_clone(sampler.sampler) };

    assert!(!cloned.is_null());

    unsafe { llama_cpp_bindings_sys::llama_sampler_free(cloned) };

    Ok(())
}

#[test]
#[serial]
fn samples_token_constrained_by_grammar() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let prompt = "Answer yes or no:";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let llg_sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let token = chain.sample(&context, batch.n_tokens() - 1)?;
    chain.accept(token)?;

    Ok(())
}

#[test]
#[serial]
fn accept_invalid_token_id_does_not_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mut sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    let huge_token = LlamaToken(i32::MAX - 1);
    let _ = sampler.accept(huge_token);

    Ok(())
}

#[test]
#[serial]
fn apply_through_chain_during_sample_does_not_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Answer:", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let llg_sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);
    let _ = chain.sample(&context, batch.n_tokens() - 1);

    Ok(())
}
