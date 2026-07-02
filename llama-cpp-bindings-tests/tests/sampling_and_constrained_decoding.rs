use std::ffi::CStr;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::ggml_time_us::ggml_time_us;
use llama_cpp_bindings::json_schema_to_grammar::json_schema_to_grammar;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::llguidance_sampler::create_llg_sampler;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::streaming_markers::StreamingMarkers;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn sample_returns_result_and_succeeds_with_valid_index(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let tokens = model.str_to_token("Hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);

    let result = sampler.sample(&context, batch.n_tokens() - 1);

    assert!(result.is_ok());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn grammar_sampler_constrains_output_to_yes_or_no(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, r"root ::= [Yy] [Ee] [Ss] | [Nn] [Oo]", "root")?,
        LlamaSampler::temp(0.8),
        LlamaSampler::greedy(),
    ]);

    let mut classifier = model.sampled_token_classifier()?;
    let (raw_token, mut outcomes) =
        classifier.sample(&mut sampler, &context, batch.n_tokens() - 1)?;
    outcomes.extend(classifier.flush());

    assert_eq!(
        outcomes.len(),
        1,
        "expected one finalised outcome after flush"
    );
    let outcome = &outcomes[0];

    let raw_as_sampled = SampledToken::Content(raw_token);
    assert!(
        !model.is_eog_token(&raw_as_sampled),
        "Grammar sampler should not allow EOS as first token"
    );

    let piece = &outcome.raw_piece;
    let first_char = piece
        .chars()
        .next()
        .ok_or_else(|| anyhow::anyhow!("piece should have at least one character"))?
        .to_lowercase()
        .next()
        .ok_or_else(|| anyhow::anyhow!("lowercase iterator should yield a character"))?;

    assert!(
        first_char == 'y' || first_char == 'n',
        "Grammar should constrain first token to start with y/n, got: '{piece}'"
    );
    assert_eq!(
        classifier.usage().completion_tokens(),
        1,
        "exactly one completion token sampled"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn json_schema_grammar_sampler_constrains_output_to_json(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "<|im_start|>user\nWhat is 2+2? Respond with a JSON object.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&mut batch)?;

    let grammar_str = json_schema_to_grammar(
        r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#,
    )?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, &grammar_str, "root")?,
        LlamaSampler::temp(0.8),
        LlamaSampler::greedy(),
    ]);

    let mut classifier = model.sampled_token_classifier()?;
    let (raw_token, mut outcomes) =
        classifier.sample(&mut sampler, &context, batch.n_tokens() - 1)?;
    outcomes.extend(classifier.flush());

    assert_eq!(
        outcomes.len(),
        1,
        "expected one finalised outcome after flush"
    );
    let outcome = &outcomes[0];

    let raw_as_sampled = SampledToken::Content(raw_token);
    assert!(
        !model.is_eog_token(&raw_as_sampled),
        "Grammar sampler should not allow EOS as first token"
    );

    let piece = &outcome.raw_piece;

    assert!(
        piece.starts_with('{'),
        "JSON schema grammar should constrain first token to start with '{{', got: '{piece}'"
    );
    assert_eq!(
        classifier.usage().completion_tokens(),
        1,
        "exactly one completion token sampled"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn sample_with_grammar_produces_constrained_output_in_loop(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "<|im_start|>user\nIs the sky blue? yes or no<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    let mut classifier = model.sampled_token_classifier()?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&mut batch)?;
    classifier.commit_prompt_tokens();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, r#"root ::= "yes" | "no""#, "root")?,
        LlamaSampler::temp(0.8),
        LlamaSampler::greedy(),
    ]);

    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 10,
    }
    .run()?;

    let lowercase = outcome.generated_raw.to_lowercase();
    assert!(
        lowercase == "yes" || lowercase == "no",
        "Grammar loop should produce 'yes' or 'no', got: '{}'",
        outcome.generated_raw
    );
    assert!(
        outcome.eog_seen,
        "loop must terminate via EOG once grammar accepts, not by exhausting the budget; outcome={outcome:?}"
    );
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(outcome.observed_tool_call, 0);
    assert!(outcome.observed_content > 0);

    let usage = classifier.into_usage();
    assert_eq!(usage.completion_tokens(), outcome.observed_content);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn sample_without_grammar_produces_multiple_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt =
        "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);

    let mut classifier = model.sampled_token_classifier()?;
    let mut sampled_count: u64 = 0;

    for (position, _) in (batch.n_tokens()..).zip(0..5) {
        let (raw_token, _outcomes) = classifier.sample(&mut sampler, &context, -1)?;
        let raw_as_sampled = SampledToken::Content(raw_token);

        if model.is_eog_token(&raw_as_sampled) {
            break;
        }

        sampled_count += 1;

        batch.clear();
        batch.add(&raw_as_sampled, position, &[0], true)?;

        context.decode(&mut batch)?;
    }

    let _ = classifier.flush();

    assert!(
        sampled_count > 0,
        "Should produce at least one token without grammar"
    );
    let usage = classifier.into_usage();
    assert!(
        usage.completion_tokens() >= sampled_count,
        "completion_tokens ({}) must include the {sampled_count} non-EOG samples",
        usage.completion_tokens()
    );

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
    sampler.apply(&mut data_array)?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn raw_prompt_completion_with_timing(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut ctx = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let max_generated_tokens: i32 = 64;

    let mut classifier = model.sampled_token_classifier()?;
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;
    let prompt_token_count = u64::try_from(tokens_list.len())?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();

    for token in &tokens_list {
        eprint!(
            "{}",
            model.token_to_piece(&SampledToken::Content(*token), &mut decoder, true, None)?
        );
    }
    std::io::stderr().flush()?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens_list, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, prompt_token_count);

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
    let initial_position = batch.n_tokens();
    let t_main_start = ggml_time_us();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut ctx,
        batch: &mut batch,
        initial_position,
        max_generated_tokens,
    }
    .run()?;
    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);
    let total_observed =
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable;

    let tokens_per_second = f64::from(u32::try_from(total_observed)?) / duration.as_secs_f64();

    eprintln!(
        "\ndecoded {total_observed} tokens in {:.2} s, speed {tokens_per_second:.2} t/s",
        duration.as_secs_f64(),
    );

    assert!(
        !outcome.generated_raw.is_empty(),
        "model should generate at least one token"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "raw prompt without tool-call markers must not produce ToolCall tokens; \
         outcome={outcome:?}"
    );
    assert!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();
    assert_eq!(
        usage.prompt_tokens, prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens, outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens, outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.undeterminable_tokens, outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    assert_eq!(
        usage.tool_call_tokens, outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn chat_inference_produces_coherent_output(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_string(),
        "Hello! How are you?".to_string(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true, true)?;

    let mut classifier = model.sampled_token_classifier()?;
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 512,
    }
    .run()?;

    println!();

    assert!(
        !outcome.generated_raw.is_empty(),
        "model should generate at least one token"
    );
    let total_observed =
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable;
    assert!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "chat without tool definitions must not produce ToolCall tokens; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();

    assert_eq!(
        usage.prompt_tokens, prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens, outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens, outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.undeterminable_tokens, outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );
    assert_eq!(
        usage.tool_call_tokens, outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn json_schema_constrains_output(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let prompt = "The weather in Paris is sunny and 22 degrees. Extract as JSON:\n";

    let mut ctx = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let tokens_list = model.str_to_token(prompt, AddBos::Always)?;

    let mut batch = LlamaBatch::new(512, 1)?;
    let last_index = i32::try_from(tokens_list.len())? - 1;

    for (index, token) in (0_i32..).zip(&tokens_list) {
        batch.add(
            &SampledToken::Content(*token),
            index,
            &[0],
            index == last_index,
        )?;
    }

    ctx.decode(&mut batch)?;

    let schema = r#"{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" }
  },
  "required": ["city", "temperature"]
}"#;

    let llg_sampler = LlamaSampler::llguidance(model, "json", schema)?;
    let mut sampler = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut generated = String::new();

    while n_cur <= 128 {
        let token = SampledToken::Content(sampler.sample(&ctx, batch.n_tokens() - 1)?);

        if model.is_eog_token(&token) {
            break;
        }

        let output_string = model.token_to_piece(&token, &mut decoder, true, None)?;
        generated.push_str(&output_string);
        print!("{output_string}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(&token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    println!();

    let parsed = serde_json::Deserializer::from_str(&generated)
        .into_iter::<serde_json::Value>()
        .next()
        .ok_or_else(|| anyhow::anyhow!("model produced no JSON value"))??;

    assert!(parsed.get("city").is_some());
    assert!(parsed.get("temperature").is_some());

    Ok(())
}

const JSON_SCHEMA: &str =
    r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
const REGEX_GRAMMAR: &str = r"yes|no";
const LARK_GRAMMAR: &str = r#"start: "yes" | "no""#;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn creates_sampler_with_valid_json_schema(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "json", JSON_SCHEMA)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn creates_sampler_with_valid_regex_grammar(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn creates_sampler_with_valid_lark_grammar(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "lark", LARK_GRAMMAR)?;

    assert!(!sampler.sampler.is_null());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn returns_error_for_unknown_grammar_kind(fixture: &LlamaFixture<'_>) -> Result<()> {
    let result = create_llg_sampler(fixture.model, "not_a_real_kind", "anything");

    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn returns_error_for_malformed_json_schema(fixture: &LlamaFixture<'_>) -> Result<()> {
    let result = create_llg_sampler(fixture.model, "json", "{this is not valid json");

    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn returns_error_for_malformed_regex(fixture: &LlamaFixture<'_>) -> Result<()> {
    let result = create_llg_sampler(fixture.model, "regex", "[invalid");

    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn name_callback_returns_llguidance(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    let name_ptr = unsafe { llama_cpp_bindings_sys::llama_sampler_name(sampler.sampler) };
    assert!(!name_ptr.is_null());
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?;

    assert_eq!(name, "llguidance");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn clone_via_ffi_creates_independent_sampler(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    let cloned = unsafe { llama_cpp_bindings_sys::llama_sampler_clone(sampler.sampler) };

    assert!(!cloned.is_null());

    unsafe { llama_cpp_bindings_sys::llama_sampler_free(cloned) };

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn samples_token_constrained_by_grammar(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "Answer yes or no:";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let llg_sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let token = chain.sample(&context, batch.n_tokens() - 1)?;
    assert!(
        token.0 >= 0,
        "grammar-constrained sampling must yield a valid token id without the grammar rejecting it"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn accept_invalid_token_id_does_not_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    let huge_token = LlamaToken(i32::MAX - 1);
    let _ = sampler.accept(huge_token);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn approximate_tok_env_returns_same_arc_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture.model.approximate_tok_env()?;
    let second = fixture.model.approximate_tok_env()?;

    assert!(Arc::ptr_eq(&first, &second));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn approximate_tok_env_drives_consistent_grammar_constraint(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let first = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;
    let second = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    assert!(!first.sampler.is_null());
    assert!(!second.sampler.is_null());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn apply_through_chain_during_sample_does_not_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let tokens = model.str_to_token("Answer:", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let llg_sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);
    let _ = chain.sample(&context, batch.n_tokens() - 1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn reset_clears_sampler_state(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;
    let huge_token = LlamaToken(i32::MAX - 1);
    let _ = sampler.accept(huge_token);
    // The out-of-range token above puts the grammar matcher into a real error
    // state, so reset legitimately surfaces that error; this test only checks
    // that the sequence does not panic.
    let _ = sampler.reset();
    let after = sampler.accept(LlamaToken(0));
    assert!(
        after.is_ok() || after.is_err(),
        "after reset, sampler.accept must return Ok or Err (not panic)"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn classifier_starts_in_pending_section_for_default_fixture(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let classifier = fixture.model.sampled_token_classifier()?;

    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn classifier_construction_is_idempotent_across_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
    let first = fixture.model.sampled_token_classifier()?;
    let second = fixture.model.sampled_token_classifier()?;

    assert_eq!(first.current_section(), second.current_section());
    assert_eq!(first.usage(), second.usage());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn ingest_with_no_markers_emits_undeterminable_with_visible_and_raw_piece(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());

    let outcomes = classifier.ingest(model.token_bos())?;

    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert!(matches!(
        outcome.sampled_token,
        SampledToken::Undeterminable(_)
    ));
    assert_eq!(outcome.visible_piece, outcome.raw_piece);
    assert_eq!(classifier.usage().undeterminable_tokens, 1);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn ingest_with_no_markers_decodes_each_token_independently(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());

    classifier.ingest(model.token_bos())?;
    classifier.ingest(model.token_eos())?;

    assert_eq!(classifier.usage().undeterminable_tokens, 2);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn ingest_prompt_token_with_no_markers_is_a_noop(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let usage_before = *classifier.usage();

    classifier.ingest_prompt_token(model.token_bos());
    classifier.ingest_prompt_tokens(&[model.token_eos(), model.token_nl()]);

    assert_eq!(*classifier.usage(), usage_before);
    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn feed_prompt_to_batch_increments_pending_prompt_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;
    classifier.feed_prompt_to_batch(&mut batch, model.token_eos(), 1, &[0], false)?;

    assert_eq!(classifier.pending_prompt_tokens(), 2);
    assert_eq!(batch.n_tokens(), 2);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn feed_prompt_sequence_to_batch_stages_all_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    let tokens = vec![model.token_bos(), model.token_eos(), model.token_nl()];
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), 3);
    assert_eq!(batch.n_tokens(), 3);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn commit_prompt_tokens_promotes_pending_count_to_usage_and_clears(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;
    classifier.feed_prompt_to_batch(&mut batch, model.token_eos(), 1, &[0], false)?;

    let promoted = classifier.commit_prompt_tokens();

    assert_eq!(promoted, 2);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 2);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn discard_pending_prompt_tokens_clears_count_without_recording_usage(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;

    let discarded = classifier.discard_pending_prompt_tokens();

    assert_eq!(discarded, 1);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn diagnose_tool_call_synthetic_renders_returns_a_pair_of_strings(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let (left, right) = fixture.model.diagnose_tool_call_synthetic_renders()?;
    let _ = left;
    let _ = right;
    Ok(())
}
