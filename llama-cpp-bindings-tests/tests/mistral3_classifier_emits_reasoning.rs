use std::num::NonZeroU32;

use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
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

const MISTRAL3_REPO: &str = "unsloth/Ministral-3-14B-Reasoning-2512-GGUF";
const MISTRAL3_FILE: &str = "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf";

const MAX_GENERATED_TOKENS: i32 = 768;

// Mistral 3 Reasoning's chat template wraps thoughts in `[THINK]...[/THINK]` and
// relies on a fine-tuned default system prompt to make the model emit them.
// Unlike Qwen3.5/3.6, Mistral does not pre-inject `[THINK]` into the generation
// prompt — the model itself emits the open marker as its first generated token.
// We craft the prompt manually rather than going through the legacy chat-template
// engine to keep the test independent of jinja-engine quirks.
const MISTRAL3_THINKING_PROMPT: &str = "\
[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\n\
First draft your thinking process (inner monologue) until you arrive at a response. \
Format your response using Markdown, and use LaTeX for any mathematical equations. \
Write both your thoughts and the response in the same language as the input.\n\n\
Your thinking process must follow the template below:\
[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. \
Be as casual and as long as you want until you are confident to generate the response \
to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT]\
[INST]Reply with the single word: four. Do not explain.[/INST]";

const FORBIDDEN_MARKERS: &[&str] = &["[THINK]", "[/THINK]"];

#[test]
fn mistral3_classifier_emits_reasoning_for_thinking_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(MISTRAL3_REPO, MISTRAL3_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(MISTRAL3_THINKING_PROMPT, AddBos::Always)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let context_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
    let mut context = model.new_context(&backend, context_params)?;

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
        bail!("Mistral 3 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(
        !outcome.generated_raw.is_empty(),
        "Mistral 3 must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "Mistral 3 classifier must emit at least one Reasoning token when the model \
         opens a [THINK] block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "Mistral 3 usage.reasoning_tokens must be non-zero when the model emits a \
         [THINK] block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Mistral 3: prompt-token replay must transition the section before generation, \
         so no Undeterminable tokens may be emitted; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Mistral 3: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "Mistral 3: completion tokens must equal observed Content + Reasoning"
    );
    assert!(
        !parsed.reasoning_content.is_empty(),
        "Mistral 3 must close its reasoning block within {MAX_GENERATED_TOKENS} tokens; \
         increase the budget or pick a more direct prompt. generated={:?}",
        outcome.generated_raw,
    );
    assert_eq!(
        outcome.reasoning_stream, parsed.reasoning_content,
        "Mistral 3: per-token reasoning stream must equal parser-side reasoning_content \
         (any difference means a marker leaked into the user-visible stream)",
    );
    assert_eq!(
        outcome.content_stream, parsed.content,
        "Mistral 3: per-token content stream must equal parser-side content \
         (any difference means a marker leaked into the user-visible stream)",
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "Mistral 3: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Mistral 3: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}
