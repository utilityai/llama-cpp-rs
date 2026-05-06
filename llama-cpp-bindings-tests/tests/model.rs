use std::num::NonZeroU16;
use std::num::NonZeroU32;
use std::path::PathBuf;

use anyhow::Result;
use llama_cpp_bindings::ChatTemplateError;
use llama_cpp_bindings::LlamaLoraAdapterInitError;
use llama_cpp_bindings::LlamaModelLoadError;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::json_schema_to_grammar;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::TestFixture;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use serial_test::serial;

#[test]
#[serial]
fn model_loads_with_valid_metadata() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.n_vocab() > 0);
    assert!(model.n_embd() > 0);
    assert!(model.n_params() > 0);
    assert!(model.n_ctx_train()? > 0);

    Ok(())
}

#[test]
#[serial]
fn special_tokens_exist() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let bos = model.token_bos();
    let eos = model.token_eos();

    assert_ne!(bos, eos);
    assert!(model.is_eog_token(&SampledToken::Content(eos)));
}

#[test]
#[serial]
fn str_to_token_roundtrip() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let tokens = model.str_to_token("hello world", AddBos::Never)?;
    assert!(!tokens.is_empty());
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let piece =
        model.token_to_piece(&SampledToken::Content(tokens[0]), &mut decoder, false, None)?;

    assert!(!piece.is_empty());

    Ok(())
}

#[test]
#[serial]
fn chat_template_returns_non_empty() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let template = model.chat_template(None);

    assert!(template.is_ok());
}

#[test]
#[serial]
fn apply_chat_template_produces_prompt() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let template = model.chat_template(None)?;
    let message = LlamaChatMessage::new("user".to_string(), "hello".to_string())?;
    let prompt = model.apply_chat_template(&template, &[message], true);

    assert!(prompt.is_ok());
    assert!(!prompt?.is_empty());

    Ok(())
}

#[test]
#[serial]
fn meta_count_returns_positive() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.meta_count() > 0);
}

#[test]
#[serial]
fn tokens_iterator_produces_valid_entries() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mut count = 0;

    for (token, _piece_result) in model.tokens(false) {
        assert!(token.0 >= 0);
        count += 1;

        if count >= 100 {
            break;
        }
    }

    assert_eq!(count, 100);
}

#[test]
#[serial]
fn token_to_piece_bytes_returns_bytes_for_known_token() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let tokens = model.str_to_token("hello", AddBos::Never)?;
    let bytes = model.token_to_piece_bytes(tokens[0], 32, false, None)?;

    assert!(!bytes.is_empty());

    Ok(())
}

#[test]
#[serial]
fn n_layer_returns_positive() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.n_layer()? > 0);

    Ok(())
}

#[test]
#[serial]
fn n_head_returns_positive() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.n_head()? > 0);

    Ok(())
}

#[test]
#[serial]
fn n_head_kv_returns_positive() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.n_head_kv()? > 0);

    Ok(())
}

#[test]
#[serial]
fn is_hybrid_returns_bool_for_test_model() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let _ = model.is_hybrid();
}

#[test]
#[serial]
fn meta_key_by_index_returns_valid_key() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let key = model.meta_key_by_index(0)?;

    assert!(!key.is_empty());

    Ok(())
}

#[test]
#[serial]
fn meta_val_str_by_index_returns_valid_value() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let value = model.meta_val_str_by_index(0)?;

    assert!(!value.is_empty());

    Ok(())
}

#[test]
#[serial]
fn meta_key_by_index_out_of_range_returns_error() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = model.meta_key_by_index(999_999);

    assert!(result.is_err());
}

#[test]
#[serial]
fn meta_val_str_by_index_out_of_range_returns_error() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = model.meta_val_str_by_index(999_999);

    assert!(result.is_err());
}

#[test]
#[serial]
fn meta_val_str_returns_value_for_known_key() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let first_key = model.meta_key_by_index(0)?;
    let value = model.meta_val_str(&first_key)?;

    assert!(!value.is_empty());

    Ok(())
}

#[test]
#[serial]
fn model_size_returns_nonzero() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(model.size() > 0);
}

#[test]
#[serial]
fn is_recurrent_returns_false_for_transformer() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    assert!(!model.is_recurrent());
}

#[test]
#[serial]
fn rope_type_does_not_panic() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let _rope_type = model.rope_type();
}

#[test]
#[serial]
fn load_model_with_invalid_path_returns_error() {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model_params = LlamaModelParams::default();
    let result = LlamaModel::load_from_file(backend, "/nonexistent/model.gguf", &model_params);

    assert_eq!(
        result.unwrap_err(),
        LlamaModelLoadError::FileNotFound(PathBuf::from("/nonexistent/model.gguf"))
    );
}

#[test]
#[serial]
fn load_model_with_invalid_file_content_returns_null_result() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model_params = LlamaModelParams::default();
    let dummy_path = std::env::temp_dir().join("llama_test_invalid_model.gguf");
    std::fs::write(&dummy_path, b"not a valid gguf model file")?;

    let result = LlamaModel::load_from_file(backend, &dummy_path, &model_params);

    assert_eq!(result.unwrap_err(), LlamaModelLoadError::NullResult);
    let _ = std::fs::remove_file(&dummy_path);

    Ok(())
}

#[cfg(unix)]
#[test]
#[serial]
fn load_model_with_non_utf8_path_returns_path_to_str_error() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model_params = LlamaModelParams::default();
    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

    let result = LlamaModel::load_from_file(backend, non_utf8_path, &model_params);

    assert_eq!(
        result.unwrap_err(),
        LlamaModelLoadError::PathToStrError(non_utf8_path.to_path_buf())
    );
}

#[cfg(unix)]
#[test]
#[serial]
fn lora_adapter_init_with_non_utf8_path_returns_error() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

    let result = model.lora_adapter_init(non_utf8_path);

    assert_eq!(
        result.unwrap_err(),
        LlamaLoraAdapterInitError::PathToStrError(non_utf8_path.to_path_buf())
    );
}

#[test]
#[serial]
fn lora_adapter_init_with_invalid_path_returns_error() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = model.lora_adapter_init("/nonexistent/path/lora.gguf");

    assert_eq!(
        result.unwrap_err(),
        LlamaLoraAdapterInitError::FileNotFound(PathBuf::from("/nonexistent/path/lora.gguf"))
    );
}

#[test]
#[serial]
fn new_context_returns_valid_context() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(256));
    let context = model.new_context(backend, ctx_params)?;

    assert!(context.n_ctx() > 0);

    Ok(())
}

#[test]
#[serial]
fn token_nl_returns_valid_token() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let nl_token = model.token_nl();

    assert!(nl_token.0 >= 0);
}

#[test]
#[serial]
fn decode_start_token_returns_valid_token() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let _decode_start = model.decode_start_token();
}

#[test]
#[serial]
fn token_sep_returns_valid_token() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let _sep_token = model.token_sep();
}

#[test]
#[serial]
fn token_to_piece_handles_large_token_requiring_buffer_resize() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    for (token, _) in model.tokens(true).take(200) {
        let result = model.token_to_piece(&SampledToken::Content(token), &mut decoder, true, None);
        assert!(result.is_ok());
    }
}

#[test]
#[serial]
fn token_to_piece_bytes_insufficient_buffer_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let tokens = model.str_to_token("hello", AddBos::Never)?;
    let result = model.token_to_piece_bytes(tokens[0], 1, false, None);

    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Insufficient Buffer Space")
    );

    Ok(())
}

#[test]
#[serial]
fn token_to_piece_with_lstrip() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let tokens = model.str_to_token("hello", AddBos::Never)?;
    let result = model.token_to_piece(
        &SampledToken::Content(tokens[0]),
        &mut decoder,
        false,
        NonZeroU16::new(1),
    );

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn n_vocab_matches_tokens_iterator_count() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let n_vocab = model.n_vocab();
    let count = model.tokens(false).count();

    assert_eq!(count, usize::try_from(n_vocab)?);

    Ok(())
}

#[test]
#[serial]
fn token_attr_returns_valid_attr() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let bos = model.token_bos();
    let _attr = model.token_attr(bos)?;

    Ok(())
}

#[test]
#[serial]
fn vocab_type_returns_valid_type() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let _vocab_type = model.vocab_type()?;

    Ok(())
}

#[test]
#[serial]
fn apply_chat_template_buffer_resize_with_long_messages() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let template = model.chat_template(None)?;
    let long_content = "a".repeat(2000);
    let message = LlamaChatMessage::new("user".to_string(), long_content)?;
    let prompt = model.apply_chat_template(&template, &[message], true);

    assert!(prompt.is_ok());
    assert!(!prompt?.is_empty());

    Ok(())
}

#[test]
#[serial]
fn meta_val_str_with_long_value_triggers_buffer_resize() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let count = model.meta_count();

    for index in 0..count {
        let key = model.meta_key_by_index(index);
        let value = model.meta_val_str_by_index(index);
        assert!(key.is_ok());
        assert!(value.is_ok());
    }
}

#[test]
#[serial]
fn str_to_token_with_add_bos_never() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let tokens_with_bos = model.str_to_token("hello", AddBos::Always)?;
    let tokens_without_bos = model.str_to_token("hello", AddBos::Never)?;

    assert!(tokens_with_bos.len() >= tokens_without_bos.len());

    Ok(())
}

#[test]
#[serial]
fn chat_template_with_nonexistent_name_returns_error() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let result = model.chat_template(Some("nonexistent_template_name_xyz"));

    assert_eq!(result.unwrap_err(), ChatTemplateError::MissingTemplate);
}

#[test]
#[serial]
fn lora_adapter_init_with_invalid_gguf_returns_null_result() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let dummy_path = std::env::temp_dir().join("llama_test_dummy_lora.gguf");
    std::fs::write(&dummy_path, b"not a valid gguf")?;

    let result = model.lora_adapter_init(&dummy_path);

    assert_eq!(result.unwrap_err(), LlamaLoraAdapterInitError::NullResult);
    let _ = std::fs::remove_file(&dummy_path);

    Ok(())
}

#[test]
#[serial]
fn str_to_token_with_many_tokens_triggers_buffer_resize() -> Result<()> {
    use std::fmt::Write;

    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let many_numbers = (0..2000).fold(String::new(), |mut accumulator, number| {
        let _ = write!(accumulator, "{number} ");
        accumulator
    });

    let tokens = model.str_to_token(&many_numbers, AddBos::Always)?;

    assert!(tokens.len() > many_numbers.len() / 2);

    Ok(())
}

#[test]
#[serial]
fn rope_type_returns_valid_result_for_test_model() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let _rope_type = model.rope_type();
}

#[test]
#[serial]
fn meta_val_str_with_null_byte_in_key_returns_error() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let result = model.meta_val_str("key\0with_null");

    assert!(result.is_err());
}

#[test]
#[serial]
fn new_context_with_huge_ctx_returns_null_error() {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(u32::MAX));

    let result = model.new_context(backend, ctx_params);

    assert!(result.is_err());
}

#[test]
#[serial]
fn sample_returns_result_and_succeeds_with_valid_index() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(256));
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

#[test]
#[serial]
fn grammar_sampler_constrains_output_to_yes_or_no() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

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

    let mut classifier = model.sampled_token_classifier();
    let (raw_token, mut outcomes) = classifier.sample(&mut sampler, &context, batch.n_tokens() - 1)?;
    outcomes.extend(classifier.flush());

    assert_eq!(outcomes.len(), 1, "expected one finalised outcome after flush");
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

#[test]
#[serial]
fn json_schema_grammar_sampler_constrains_output_to_json() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

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

    let mut classifier = model.sampled_token_classifier();
    let (raw_token, mut outcomes) = classifier.sample(&mut sampler, &context, batch.n_tokens() - 1)?;
    outcomes.extend(classifier.flush());

    assert_eq!(outcomes.len(), 1, "expected one finalised outcome after flush");
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

#[test]
#[serial]
fn sample_with_grammar_produces_constrained_output_in_loop() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let prompt = "<|im_start|>user\nIs the sky blue? yes or no<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    let mut classifier = model.sampled_token_classifier();
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
        "loop must terminate via EOG once grammar accepts, not by exhausting the budget; \
         outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "closed-think prompt must not produce Reasoning tokens; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "prompt-token replay closes the think block before generation, so the section \
         must be Content and no Undeterminable tokens may be emitted; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "prompt without tool definitions must not produce ToolCall tokens; outcome={outcome:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "grammar must yield at least one Content token (the answer); outcome={outcome:?}"
    );

    let usage = classifier.into_usage();
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "for the closed-think grammar prompt, completion_tokens equals observed Content"
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "usage.undeterminable_tokens must be zero; usage={usage:?}"
    );

    Ok(())
}

#[test]
#[serial]
fn sample_without_grammar_produces_multiple_tokens() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let prompt =
        "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);

    let mut classifier = model.sampled_token_classifier();
    let mut sampled_count: u64 = 0;
    let mut position = batch.n_tokens();

    for _ in 0..5 {
        let (raw_token, _outcomes) = classifier.sample(&mut sampler, &context, -1)?;
        let raw_as_sampled = SampledToken::Content(raw_token);

        if model.is_eog_token(&raw_as_sampled) {
            break;
        }

        sampled_count += 1;

        batch.clear();
        batch.add(&raw_as_sampled, position, &[0], true)?;
        position += 1;

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
