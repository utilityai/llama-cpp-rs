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

    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();

    for token in &tokens_list {
        eprint!(
            "{}",
            model.token_to_piece(&SampledToken::Content(*token), &mut decoder, true, None)?
        );
    }
    std::io::stderr().flush()?;

    let mut batch = LlamaBatch::new(512, 1)?;
    let last_index = i32::try_from(tokens_list.len() - 1)?;

    for (index, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = index == last_index;
        batch.add(&SampledToken::Content(token), index, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let mut n_cur = batch.n_tokens();
    let mut n_decode: i32 = 0;
    let t_main_start = ggml_time_us();

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

    let mut generated = String::new();

    while n_cur <= n_len {
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

    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    let last_index = i32::try_from(tokens.len())? - 1;
    for (position, token) in (0_i32..).zip(tokens.into_iter()) {
        let output_logits = position == last_index;
        batch.add(&SampledToken::Content(token), position, &[0], output_logits)?;
    }
    context.decode(&mut batch)?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let mut classifier = model.reasoning_token_classifier()?;
    let mut position = batch.n_tokens();
    let max_tokens = 1024;
    let mut generated = String::new();
    let mut saw_reasoning = false;
    let mut saw_content = false;

    while position <= max_tokens {
        let token = classifier.classify(sampler.sample(&context, batch.n_tokens() - 1)?);

        saw_reasoning |= matches!(token, SampledToken::Reasoning(_));
        saw_content |= matches!(token, SampledToken::Content(_));

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
        saw_reasoning,
        "expected at least one Reasoning token from the classifier"
    );
    assert!(
        saw_content,
        "expected at least one Content token from the classifier"
    );

    Ok(())
}
