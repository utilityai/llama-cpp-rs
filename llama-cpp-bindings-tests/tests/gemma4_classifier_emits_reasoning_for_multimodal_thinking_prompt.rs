use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::test_model::fixtures_dir;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

const MAX_GENERATED_TOKENS: i32 = 200;

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma4_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker();
    let prompt = format!(
        "<bos><start_of_turn>user\n{marker}What animals do you see in this image?<end_of_turn>\n<start_of_turn>model\n<|channel>thought\n"
    );

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier = model.sampled_token_classifier();
    let n_past = classifier.eval_multimodal_chunks(&chunks, mtmd_ctx, &context, 0, 0, 512, true)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);

    let mut batch = LlamaBatch::new(2048, 1)?;
    let outcome = ClassifySampleLoop {
        model,
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
            "Gemma 4 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the prompt opens a `<|channel>thought` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        anyhow::bail!(
            "Gemma 4 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}

llama_tests_main!();
