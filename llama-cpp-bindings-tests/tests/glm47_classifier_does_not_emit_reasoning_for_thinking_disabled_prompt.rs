use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const GLM47_REPO: &str = "unsloth/GLM-4.7-Flash-GGUF";
const GLM47_FILE: &str = "GLM-4.7-Flash-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 200;

// GLM-4.7-Flash with reasoning disabled: the chat template renders a closed
// `</think>` immediately after `<|assistant|>\n`, leaving the model outside
// the reasoning section before generation begins. No reasoning tokens should
// ever be classified.
const GLM47_THINKING_DISABLED_PROMPT: &str = "\
<|user|>
What is 2 + 2?
<|assistant|>
</think>

";

const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

#[test]
fn glm47_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(GLM47_REPO, GLM47_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GLM47_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let context_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
    let mut context = model.new_context(&backend, context_params)?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model: &model,
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
        "GLM-4.7: must generate at least one token"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "GLM-4.7 thinking-disabled: classifier must not emit any Reasoning token \
         when the prompt closes the think block before generation begins; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "GLM-4.7 thinking-disabled: prompt-token replay must move section to Content \
         before generation, so no Undeterminable tokens may be emitted; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "GLM-4.7 thinking-disabled: usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "GLM-4.7 thinking-disabled: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "GLM-4.7 thinking-disabled: classifier must emit at least one Content token"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "GLM-4.7 thinking-disabled: completion tokens must equal observed Content tokens"
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.content_stream.contains(forbidden),
            "GLM-4.7 thinking-disabled: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
