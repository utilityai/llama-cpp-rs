use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const QWEN36_REPO: &str = "unsloth/Qwen3.6-35B-A3B-GGUF";
const QWEN36_FILE: &str = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

#[test]
fn qwen36_chat_inference_emits_reasoning_when_template_auto_opens() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(QWEN36_REPO, QWEN36_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let context_params = LlamaContextParams::default();
    let mut context = LlamaContext::from_model(&model, &backend, context_params)?;

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
        model: &model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 1024,
    }
    .run()?;

    assert!(
        !outcome.generated_raw.is_empty(),
        "Qwen3.6 must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "Qwen3.6 chat template auto-opens reasoning, so the classifier must emit at \
         least one Reasoning token; outcome={outcome:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "Qwen3.6 must emit at least one Content token after </think>; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Qwen3.6 chat template auto-opens reasoning, so the classifier must never emit \
         Undeterminable; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "chat without tool definitions must not produce ToolCall tokens; outcome={outcome:?}"
    );

    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.6 chat template must be recognised by the parser; got Unrecognized");
    };
    assert!(
        !parsed.content.is_empty(),
        "parser must see post-</think> content in generated text; generated={:?}",
        outcome.generated_raw
    );

    let usage = classifier.into_usage();
    assert_eq!(
        usage.prompt_tokens, prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.reasoning_tokens, outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Qwen3.6 with auto-opening reasoning must never produce Undeterminable"
    );

    Ok(())
}
