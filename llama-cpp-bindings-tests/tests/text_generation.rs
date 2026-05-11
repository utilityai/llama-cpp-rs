use std::io::Write;
use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::ggml_time_us;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::FixtureSession;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;

#[test]
fn raw_prompt_completion_with_timing() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(backend, ctx_params)
        .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let max_generated_tokens: i32 = 64;

    let mut classifier = model.sampled_token_classifier();
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

    #[allow(clippy::cast_precision_loss)]
    let tokens_per_second = (outcome.observed_undeterminable as f32
        + outcome.observed_content as f32
        + outcome.observed_reasoning as f32)
        / duration.as_secs_f32();

    eprintln!(
        "\ndecoded {} tokens in {:.2} s, speed {tokens_per_second:.2} t/s",
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable,
        duration.as_secs_f32(),
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
    // The raw prompt carries no chat-template markers, so the classifier starts
    // in Pending. The exact split between Content / Reasoning / Undeterminable
    // depends on the model: Qwen 3.5 keeps generating raw text and never emits
    // `<think>`, so every token is Undeterminable; Qwen 3.6 was trained to
    // start every reply with a `<think>...</think>` block even without a
    // chat template, so the same prompt yields a mix. Both behaviours are
    // correct — we only assert internal consistency below.
    let total_observed =
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable;
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

#[test]
fn chat_inference_produces_coherent_output() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();

    let context_params = LlamaContextParams::default();
    let mut context = model.new_context(backend, context_params)?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_string(),
        "Hello! How are you?".to_string(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let mut classifier = model.sampled_token_classifier();
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
        max_generated_tokens: 1024,
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
