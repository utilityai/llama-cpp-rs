#![cfg(feature = "multimodal_capable")]

use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;
use llama_cpp_bindings_tests::test_model::fixtures_dir;

const MISTRAL3_REPO: &str = "unsloth/Ministral-3-14B-Reasoning-2512-GGUF";
const MISTRAL3_FILE: &str = "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf";
const MISTRAL3_MMPROJ_FILE: &str = "mmproj-F16.gguf";

const MAX_GENERATED_TOKENS: i32 = 768;

#[test]
fn mistral3_classifier_emits_reasoning_for_multimodal_thinking_prompt() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let model_path = download_file_from(MISTRAL3_REPO, MISTRAL3_FILE)?;
    let mmproj_path = download_file_from(MISTRAL3_REPO, MISTRAL3_MMPROJ_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)?;

    let mtmd_params = MtmdContextParams::default();
    let mmproj_str = mmproj_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("mmproj path is not valid UTF-8"))?;
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)?;

    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(8192))
        .with_n_batch(512);
    let mut context = model.new_context(&backend, context_params)?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker();
    let prompt = format!(
        "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\n\
         First draft your thinking process (inner monologue) until you arrive at a response. \
         Format your response using Markdown, and use LaTeX for any mathematical equations. \
         Write both your thoughts and the response in the same language as the input.\n\n\
         Your thinking process must follow the template below:\
         [THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. \
         Be as casual and as long as you want until you are confident to generate the response \
         to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT]\
         [INST]{marker}What animals do you see in this image?[/INST]"
    );

    let input_text = MtmdInputText {
        text: prompt,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier = model.sampled_token_classifier();
    let n_past =
        classifier.eval_multimodal_chunks(&chunks, &mtmd_ctx, &context, 0, 0, 512, true)?;

    let mut sampler = LlamaSampler::greedy();
    let mut batch = LlamaBatch::new(2048, 1)?;
    let outcome = ClassifySampleLoop {
        model: &model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    if outcome.observed_reasoning == 0 {
        anyhow::bail!(
            "Mistral 3 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the model opens a `[THINK]` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        anyhow::bail!(
            "Mistral 3 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}
