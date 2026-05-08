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

const DEEPSEEK_R1_8B_REPO: &str = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF";
const DEEPSEEK_R1_8B_FILE: &str = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 1500;

// DeepSeek-R1-Distill-Llama-8B uses `<think>...</think>` reasoning markers
// and full-width-bar role tokens `<｜User｜>` / `<｜Assistant｜>` (U+FF5C,
// not ASCII `|`). The chat template's `add_generation_prompt` ALWAYS appends
// `<｜Assistant｜><think>\n` — DeepSeek-R1 is a pure reasoner with no
// thinking-disabled mode — so the model resumes generation already inside
// the reasoning block.
const DEEPSEEK_R1_8B_THINKING_PROMPT: &str = "\
<｜User｜>What is 2 + 2?<｜Assistant｜><think>
";

const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

#[test]
fn deepseek_r1_8b_classifier_emits_reasoning_for_thinking_enabled_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(DEEPSEEK_R1_8B_REPO, DEEPSEEK_R1_8B_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(DEEPSEEK_R1_8B_THINKING_PROMPT, AddBos::Never)?;
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
    let parsed = model.parse_chat_message("[]", &outcome.generated_raw, false)?;

    assert!(
        !outcome.generated_raw.is_empty(),
        "DeepSeek-R1-8B: must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "DeepSeek-R1-8B: classifier must emit at least one Reasoning token when the prompt \
         opens a <think> block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "DeepSeek-R1-8B: usage.reasoning_tokens must be non-zero when the prompt opens a \
         <think> block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "DeepSeek-R1-8B: prompt-token replay must move section to Reasoning before generation, \
         so no Undeterminable tokens may be emitted; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "DeepSeek-R1-8B: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "DeepSeek-R1-8B: completion tokens must equal observed Content + Reasoning"
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!(
            "DeepSeek-R1-8B didn't close its reasoning block within {MAX_GENERATED_TOKENS} \
             tokens — skipping strict parser-equality assertions"
        );
    } else {
        assert_eq!(
            outcome.reasoning_stream, parsed.reasoning_content,
            "DeepSeek-R1-8B: per-token reasoning stream must equal parser-side reasoning_content \
             (any difference means a marker leaked into the user-visible stream)",
        );
        assert_eq!(
            outcome.content_stream, parsed.content,
            "DeepSeek-R1-8B: per-token content stream must equal parser-side content \
             (any difference means a marker leaked into the user-visible stream)",
        );
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "DeepSeek-R1-8B: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "DeepSeek-R1-8B: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
