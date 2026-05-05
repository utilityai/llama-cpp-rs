use std::io::Write;
use std::time::Duration;

use anyhow::{Context, Result};
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::ggml_time_us;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::{AddBos, LlamaChatMessage};
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::TestFixture;

#[test]
fn raw_prompt_completion_with_timing() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(backend, ctx_params)
        .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let n_len: i32 = 64;

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
    assert_eq!(classifier.usage().prompt_tokens(), 0);

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens(), prompt_token_count);

    let mut n_cur = batch.n_tokens();
    let mut n_decode: i32 = 0;
    let mut observed_content: u64 = 0;
    let mut observed_reasoning: u64 = 0;
    let t_main_start = ggml_time_us();

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

    let mut generated = String::new();

    while n_cur <= n_len {
        let token = classifier.sample(&mut sampler, &ctx, batch.n_tokens() - 1)?;

        match token {
            SampledToken::Content(_) => observed_content += 1,
            SampledToken::Reasoning(_) => observed_reasoning += 1,
            SampledToken::ToolCall(_) => {}
            SampledToken::Undeterminable(_) => {}
        }

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

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;
        n_decode += 1;
    }

    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

    #[allow(clippy::cast_precision_loss)]
    let tokens_per_second = n_decode as f32 / duration.as_secs_f32();

    eprintln!(
        "\ndecoded {n_decode} tokens in {:.2} s, speed {tokens_per_second:.2} t/s",
        duration.as_secs_f32(),
    );

    assert!(
        !generated.is_empty(),
        "model should generate at least one token"
    );

    let usage = classifier.into_usage();
    assert_eq!(
        usage.prompt_tokens(),
        prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens(),
        observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens(),
        observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        observed_content + observed_reasoning
    );

    Ok(())
}

#[test]
fn chat_inference_produces_coherent_output() -> Result<()> {
    let fixture = TestFixture::shared();
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

    let mut classifier = model.sampled_token_classifier()?;
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens(), 0);

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let mut position = batch.n_tokens();
    let max_tokens = 1024;
    let mut generated = String::new();
    let mut observed_content: u64 = 0;
    let mut observed_reasoning: u64 = 0;

    while position <= max_tokens {
        let token = classifier.sample(&mut sampler, &context, batch.n_tokens() - 1)?;

        match token {
            SampledToken::Content(_) => observed_content += 1,
            SampledToken::Reasoning(_) => observed_reasoning += 1,
            SampledToken::ToolCall(_) => {}
            SampledToken::Undeterminable(_) => {
                unreachable!(
                    "Qwen3 chat template uses detected reasoning markers; classifier must not emit Undeterminable"
                )
            }
        }

        if model.is_eog_token(&token) {
            break;
        }

        let piece = model.token_to_piece(&token, &mut decoder, true, None)?;
        generated.push_str(&piece);
        print!("{piece}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(&token, position, &[0], true)?;
        position += 1;

        context.decode(&mut batch)?;
    }

    println!();

    assert!(
        !generated.is_empty(),
        "model should generate at least one token"
    );
    assert!(
        observed_reasoning > 0,
        "reasoning model should emit at least one Reasoning token"
    );
    assert!(
        observed_content > 0,
        "reasoning model should emit at least one Content token after </think>"
    );

    let usage = classifier.into_usage();

    assert_eq!(
        usage.prompt_tokens(),
        prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens(),
        observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens(),
        observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        observed_content + observed_reasoning
    );
    assert_eq!(
        usage.undeterminable_tokens(),
        0,
        "model with detected markers should never produce Undeterminable"
    );

    Ok(())
}
