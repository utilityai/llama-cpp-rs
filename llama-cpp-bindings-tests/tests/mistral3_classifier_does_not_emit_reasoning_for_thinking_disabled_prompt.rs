use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

const MAX_GENERATED_TOKENS: i32 = 200;

const MISTRAL3_THINKING_DISABLED_PROMPT: &str = "\
[INST]Reply with the single word: four. Do not explain.[/INST][THINK][/THINK]";

const FORBIDDEN_MARKERS: &[&str] = &["[THINK]", "[/THINK]"];

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn mistral3_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(MISTRAL3_THINKING_DISABLED_PROMPT, AddBos::Always)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

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
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(!outcome.generated_raw.is_empty());
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(usage.completion_tokens(), outcome.observed_content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

llama_tests_main!();
