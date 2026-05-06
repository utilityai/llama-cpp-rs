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

const QWEN36_REPO: &str = "unsloth/Qwen3.6-35B-A3B-GGUF";
const QWEN36_FILE: &str = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 1500;

// Qwen3.6's chat template injects `<think>\n` directly into the generation prompt
// when `enable_thinking=true` (the default). The legacy `llama_chat_apply_template`
// path bypasses that jinja branch, so we craft the prompt manually to faithfully
// reproduce the production case where the model resumes generation already inside
// the reasoning block.
const QWEN36_THINKING_PROMPT: &str = "\
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>
";

const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

#[test]
fn qwen36_classifier_emits_reasoning_for_thinking_enabled_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(QWEN36_REPO, QWEN36_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(QWEN36_THINKING_PROMPT, AddBos::Never)?;
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
    let parsed = model.parse_chat_message("[]", &outcome.generated_raw, true)?;

    assert!(
        !outcome.generated_raw.is_empty(),
        "Qwen3.6 must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "Qwen3.6: classifier must emit at least one Reasoning token when the prompt \
         opens a <think> block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "Qwen3.6: usage.reasoning_tokens must be non-zero when the prompt opens a \
         <think> block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Qwen3.6: prompt-token replay must move section to Reasoning before generation, \
         so no Undeterminable tokens may be emitted; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Qwen3.6: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "Qwen3.6: completion tokens must equal observed Content + Reasoning"
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!(
            "Qwen3.6 parser returned empty reasoning_content (likely a partial parse \
             over `<|im_end|>`-truncated output) — relying on the FORBIDDEN_MARKERS \
             substring check below for leak detection."
        );
    } else {
        assert_eq!(
            outcome.reasoning_stream, parsed.reasoning_content,
            "Qwen3.6: per-token reasoning stream must equal parser-side reasoning_content \
             (any difference means a marker leaked into the user-visible stream)",
        );
        assert_eq!(
            outcome.content_stream, parsed.content,
            "Qwen3.6: per-token content stream must equal parser-side content \
             (any difference means a marker leaked into the user-visible stream)",
        );
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "Qwen3.6: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Qwen3.6: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
