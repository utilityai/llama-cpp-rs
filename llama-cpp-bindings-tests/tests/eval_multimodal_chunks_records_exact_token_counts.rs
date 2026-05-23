use anyhow::Result;
use llama_cpp_bindings::TokenUsage;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputChunks;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings_tests::test_model::fixtures_dir;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

const PROMPT_QUESTION: &str = "What animals do you see in this image?";

struct ExpectedChunkTotals {
    text: u64,
    image: u64,
    audio: u64,
}

fn sum_chunk_token_counts_by_type(chunks: &MtmdInputChunks) -> Result<ExpectedChunkTotals> {
    let mut totals = ExpectedChunkTotals {
        text: 0,
        image: 0,
        audio: 0,
    };
    for index in 0..chunks.len() {
        let chunk = chunks
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("chunk index {index} should exist"))?;
        let n_tokens = u64::try_from(chunk.n_tokens())?;
        match chunk.chunk_type()? {
            MtmdInputChunkType::Text => {
                totals.text = totals.text.saturating_add(n_tokens);
            }
            MtmdInputChunkType::Image => {
                totals.image = totals.image.saturating_add(n_tokens);
            }
            MtmdInputChunkType::Audio => {
                totals.audio = totals.audio.saturating_add(n_tokens);
            }
        }
    }
    Ok(totals)
}

fn build_multimodal_chunks_and_eval_into_usage(
    fixture: &LlamaFixture<'_>,
) -> Result<(TokenUsage, ExpectedChunkTotals)> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker();
    let prompt = format!("{marker}{PROMPT_QUESTION}");

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let expected = sum_chunk_token_counts_by_type(&chunks)?;

    let context_params = (*fixture.context_params).into_llama_context_params();
    let context = LlamaContext::from_model(model, fixture.backend, context_params)?;

    let mut classifier = model.sampled_token_classifier();
    classifier.eval_multimodal_chunks(&chunks, mtmd_ctx, &context, 0, 0, 512, true)?;

    Ok((classifier.into_usage(), expected))
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn prompt_tokens_match_text_chunk_total(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.prompt_tokens != expected.text {
        anyhow::bail!(
            "prompt_tokens must equal sum of text-chunk n_tokens; expected {}, got {}",
            expected.text,
            usage.prompt_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn input_image_tokens_match_image_chunk_total(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.input_image_tokens != expected.image {
        anyhow::bail!(
            "input_image_tokens must equal sum of image-chunk n_tokens; expected {}, got {}",
            expected.image,
            usage.input_image_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn input_audio_tokens_are_zero_for_image_only_input(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if expected.audio != 0 {
        anyhow::bail!(
            "fixture invariant: image-only multimodal input should produce zero audio chunk tokens, got {}",
            expected.audio
        );
    }
    if usage.input_audio_tokens != 0 {
        anyhow::bail!(
            "input_audio_tokens must be zero when no audio chunks are evaluated; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn completion_tokens_are_zero_after_eval_before_generation(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let (usage, _expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.completion_tokens() != 0 {
        anyhow::bail!(
            "completion_tokens must be zero immediately after eval (no generation has occurred); got {}",
            usage.completion_tokens()
        );
    }

    Ok(())
}

llama_tests_main!();
