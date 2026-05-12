use std::num::NonZeroU32;

use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
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

const GEMMA4_REPO: &str = "unsloth/gemma-4-E4B-it-GGUF";
const GEMMA4_FILE: &str = "gemma-4-E4B-it-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 1500;

// Gemma 4 uses asymmetric reasoning markers: `<|channel>thought` opens
// the thinking block and `<channel|>` closes it. We pre-inject the
// `<|channel>thought\n` opener at the model turn so the classifier sees
// the marker via prompt-token replay and starts generation in `Reasoning`,
// matching the behaviour of Qwen3.5/3.6's auto-injected `<think>\n`.
const GEMMA4_THINKING_PROMPT: &str = "\
<bos><start_of_turn>user\nReply with the single word: four. Do not explain.<end_of_turn>\n\
<start_of_turn>model\n<|channel>thought\n";

const FORBIDDEN_MARKERS: &[&str] = &["<|channel>thought", "<channel|>"];

#[test]
fn gemma4_classifier_emits_reasoning_for_thinking_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(GEMMA4_REPO, GEMMA4_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GEMMA4_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let context_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
    let mut context = LlamaContext::from_model(&model, &backend, context_params)?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
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
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Gemma 4 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(
        !outcome.generated_raw.is_empty(),
        "Gemma 4 must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "Gemma 4 classifier must emit at least one Reasoning token when the model \
         emits a `<|channel>thought` block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "Gemma 4 usage.reasoning_tokens must be non-zero when the model emits a \
         reasoning block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Gemma 4: classifier must not emit Undeterminable when the model emits a \
         detected `<|channel>thought` marker; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Gemma 4: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "Gemma 4: completion tokens must equal observed Content + Reasoning"
    );
    assert!(
        !parsed.reasoning_content.is_empty(),
        "Gemma 4 must close its reasoning block within {MAX_GENERATED_TOKENS} tokens; \
         increase the budget or pick a more direct prompt. generated={:?}",
        outcome.generated_raw,
    );

    // Gemma 4 goes through llama.cpp's specialized-template path, which leaves the
    // raw `<|channel>thought` prefix in `parsed.reasoning_content` rather than
    // stripping it like the differential autoparser does for Qwen3-family. So the
    // parser-equality cross-check would require a per-template carve-out — instead,
    // rely on the FORBIDDEN_MARKERS substring check below: the streams the user
    // actually sees must not contain marker text, regardless of what the parser
    // chose to keep.
    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "Gemma 4: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Gemma 4: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
