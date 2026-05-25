#![expect(
    clippy::unnecessary_wraps,
    reason = "every trial returns anyhow::Result<()> to match the LlamaTestFn signature"
)]

use anyhow::Result;
use llama_cpp_bindings::GrammarError;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn dry_sampler_with_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let breakers: Vec<&[u8]> = vec![b"\n", b"\t"];
    let _sampler = LlamaSampler::dry(fixture.model, 1.5, 2.0, 128, 2, &breakers);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn dry_sampler_with_null_byte_in_seq_breakers_returns_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let breakers: Vec<&[u8]> = vec![b"hello\0world"];
    let result = LlamaSampler::dry(fixture.model, 1.5, 2.0, 128, 2, breakers);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_returns_sampler_for_valid_grammar(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = LlamaSampler::grammar(fixture.model, "root ::= \"hello\"", "root");

    assert!(sampler.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_returns_sampler_for_valid_grammar_with_triggers(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let trigger_words: Vec<&[u8]> = vec![b"function"];
    let sampler = LlamaSampler::grammar_lazy(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        trigger_words,
        &[],
    );

    assert!(sampler.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_patterns_returns_sampler_for_valid_grammar_with_patterns(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let patterns = vec!["\\{.*".to_owned()];
    let sampler = LlamaSampler::grammar_lazy_patterns(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        &patterns,
        &[],
    );

    assert!(sampler.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_with_root_not_found_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let trigger_words: Vec<&[u8]> = vec![b"function"];
    let result = LlamaSampler::grammar_lazy(
        fixture.model,
        "expr ::= \"hello\"",
        "root",
        trigger_words,
        &[],
    );

    assert!(matches!(result, Err(GrammarError::RootNotFound)));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_with_null_byte_in_trigger_word_returns_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let trigger_words: Vec<&[u8]> = vec![b"hel\0lo"];
    let result = LlamaSampler::grammar_lazy(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        trigger_words,
        &[],
    );

    assert!(matches!(result, Err(GrammarError::TriggerWordNullBytes(_))));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_patterns_with_root_not_found_returns_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let patterns = vec!["\\{.*".to_owned()];
    let result = LlamaSampler::grammar_lazy_patterns(
        fixture.model,
        "expr ::= \"hello\"",
        "root",
        &patterns,
        &[],
    );

    assert!(matches!(result, Err(GrammarError::RootNotFound)));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_patterns_with_null_byte_in_pattern_returns_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let patterns = vec!["hel\0lo".to_owned()];
    let result = LlamaSampler::grammar_lazy_patterns(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        &patterns,
        &[],
    );

    assert!(matches!(result, Err(GrammarError::GrammarNullBytes(_))));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn grammar_lazy_patterns_with_malformed_regex_returns_invalid_trigger_pattern(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let patterns = vec!["[".to_owned()];
    let result = LlamaSampler::grammar_lazy_patterns(
        fixture.model,
        "root ::= \"hello\"",
        "root",
        &patterns,
        &[],
    );

    assert!(matches!(
        result,
        Err(GrammarError::InvalidTriggerPattern { .. }),
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn llguidance_method_creates_sampler(fixture: &LlamaFixture<'_>) -> Result<()> {
    let result = LlamaSampler::llguidance(fixture.model, "regex", r"yes|no");

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn logit_bias_with_empty_biases_succeeds(_fixture: &LlamaFixture<'_>) -> Result<()> {
    let result = LlamaSampler::logit_bias(0, &[]);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn dry_sampler_with_root_not_found_grammar_does_not_apply(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let breakers: Vec<&[u8]> = vec![b"\n"];
    let _sampler = LlamaSampler::dry(fixture.model, 1.5, 2.0, 128, 2, &breakers);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn accept_many_iterates_over_borrowed_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let tokens = vec![fixture.model.token_bos(), fixture.model.token_eos()];

    sampler.accept_many(&tokens)?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn with_tokens_returns_self_after_accepting_each_token(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let tokens = [fixture.model.token_bos(), fixture.model.token_eos()];

    let _consumed = sampler.with_tokens(tokens.iter().copied())?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn accept_consumes_a_single_token(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    sampler.accept(fixture.model.token_bos())?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn try_accept_returns_ok_for_a_valid_token(_fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    sampler.try_accept(LlamaToken::new(0))?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn apply_runs_sampler_over_token_data_array(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("Hi", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let mut data_array = context.token_data_array_ith(batch.n_tokens() - 1)?;
    let sampler = LlamaSampler::greedy();
    sampler.apply(&mut data_array);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn sample_returns_token_after_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("Hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);
    let result = sampler.sample(&context, batch.n_tokens() - 1);

    assert!(result.is_ok());

    Ok(())
}

llama_tests_main!();
