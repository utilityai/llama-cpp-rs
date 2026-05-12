use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
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

const DEEPSEEK_R1_8B_REPO: &str = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF";
const DEEPSEEK_R1_8B_FILE: &str = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 200;

// DeepSeek-R1-Distill-Llama-8B has no native thinking-disabled mode in its
// chat template (R1 is a pure reasoner). This prompt manually closes the
// `<think>` block before generation so the classifier starts in CONTENT —
// verifies the "spurious close in content section" path with this model's
// tokenizer and still produces zero Reasoning tokens.
const DEEPSEEK_R1_8B_THINKING_DISABLED_PROMPT: &str = "\
<｜User｜>What is 2 + 2?<｜Assistant｜><think>

</think>

";

const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

#[test]
fn deepseek_r1_8b_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(DEEPSEEK_R1_8B_REPO, DEEPSEEK_R1_8B_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens =
        model.str_to_token(DEEPSEEK_R1_8B_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let context_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
    let mut context = LlamaContext::from_model(&model, &backend, context_params)?;

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
        "DeepSeek-R1-8B: must generate at least one token"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "DeepSeek-R1-8B thinking-disabled: classifier must not emit any Reasoning token \
         when the prompt closes the think block before generation begins; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "DeepSeek-R1-8B thinking-disabled: prompt-token replay must move section to Content \
         before generation, so no Undeterminable tokens may be emitted; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "DeepSeek-R1-8B thinking-disabled: usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "DeepSeek-R1-8B thinking-disabled: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "DeepSeek-R1-8B thinking-disabled: classifier must emit at least one Content token"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "DeepSeek-R1-8B thinking-disabled: completion tokens must equal observed Content tokens"
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.content_stream.contains(forbidden),
            "DeepSeek-R1-8B thinking-disabled: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
