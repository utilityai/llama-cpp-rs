use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen35_chat_inference_emits_reasoning_when_template_auto_opens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_owned(),
        "Hello! How are you?".to_owned(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let mut classifier = model.sampled_token_classifier();
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

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

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(outcome.observed_tool_call, 0);

    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.5 chat template must be recognised by the parser; got Unrecognized");
    };
    assert!(!parsed.content.is_empty());

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, prompt_token_count);
    assert_eq!(usage.reasoning_tokens, outcome.observed_reasoning);
    assert_eq!(usage.undeterminable_tokens, 0);

    Ok(())
}

llama_tests_main!();
