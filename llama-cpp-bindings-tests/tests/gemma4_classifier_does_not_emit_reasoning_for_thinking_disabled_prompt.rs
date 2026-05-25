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

const GEMMA4_THINKING_DISABLED_PROMPT: &str = "\
<bos><start_of_turn>user\nReply with the single word: four. Do not explain.<end_of_turn>\n\
<start_of_turn>model\n<|channel>thought\n<channel|>\n";

const FORBIDDEN_MARKERS: &[&str] = &["<|channel>thought", "<channel|>"];

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn gemma4_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GEMMA4_THINKING_DISABLED_PROMPT, AddBos::Never)?;
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

    assert!(
        !outcome.generated_raw.is_empty(),
        "Gemma 4 must generate at least one token"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "Gemma 4 thinking-disabled: classifier must not emit any Reasoning token \
         when the prompt closes the thought channel before generation begins; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Gemma 4 thinking-disabled: prompt-token replay must move section to Content \
         before generation, so no Undeterminable tokens may be emitted; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "Gemma 4 thinking-disabled: usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Gemma 4 thinking-disabled: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "Gemma 4 thinking-disabled: classifier must emit at least one Content token"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "Gemma 4 thinking-disabled: completion tokens must equal observed Content tokens"
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Gemma 4 thinking-disabled: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}

llama_tests_main!();
