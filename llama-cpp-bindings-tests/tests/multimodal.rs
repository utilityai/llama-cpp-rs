use anyhow::{Context, Result};
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_bindings::mtmd::{MtmdBitmap, MtmdInputChunkType, MtmdInputChunks, MtmdInputText};
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_tests::test_model;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

struct ChunkTokenBreakdown {
    text: u64,
    image: u64,
    audio: u64,
}

fn count_chunk_tokens_by_type(chunks: &MtmdInputChunks) -> Result<ChunkTokenBreakdown> {
    let mut breakdown = ChunkTokenBreakdown {
        text: 0,
        image: 0,
        audio: 0,
    };
    for index in 0..chunks.len() {
        let chunk = chunks
            .get(index)
            .with_context(|| format!("chunk index {index} is missing"))?;
        let n_tokens = u64::try_from(chunk.n_tokens())?;
        match chunk.chunk_type()? {
            MtmdInputChunkType::Text => breakdown.text += n_tokens,
            MtmdInputChunkType::Image => breakdown.image += n_tokens,
            MtmdInputChunkType::Audio => breakdown.audio += n_tokens,
        }
    }

    Ok(breakdown)
}

fn build_user_prompt_with_image_marker(model: &LlamaModel, question: &str) -> Result<String> {
    let marker = llama_cpp_bindings::mtmd::mtmd_default_marker();
    let user_content = format!("{marker}{question}");
    let chat_template = model.chat_template(None)?;
    let messages = [LlamaChatMessage::new("user".to_string(), user_content)?];

    Ok(model.apply_chat_template(&chat_template, &messages, true)?)
}

struct SamplingTotals {
    generated: String,
    observed_content: u64,
    observed_reasoning: u64,
}

fn drive_sampling_loop(
    classifier: &mut SampledTokenClassifier,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    starting_position: llama_pos,
    max_tokens: usize,
) -> Result<SamplingTotals> {
    let mut sampler = LlamaSampler::greedy();
    let mut totals = SamplingTotals {
        generated: String::new(),
        observed_content: 0,
        observed_reasoning: 0,
    };
    let mut batch = LlamaBatch::new(512, 1)?;
    let mut current_position = starting_position;

    for _ in 0..max_tokens {
        let (raw_token, outcomes) = classifier.sample(&mut sampler, ctx, -1)?;
        for outcome in &outcomes {
            totals.generated.push_str(&outcome.raw_piece);
            match outcome.sampled_token {
                SampledToken::Content(_) => totals.observed_content += 1,
                SampledToken::Reasoning(_) => totals.observed_reasoning += 1,
                SampledToken::ToolCall(_) | SampledToken::Undeterminable(_) => {}
            }
        }

        let raw_as_sampled = SampledToken::Content(raw_token);
        if model.is_eog_token(&raw_as_sampled) {
            break;
        }

        batch.clear();
        batch.add(&raw_as_sampled, current_position, &[0], true)?;
        current_position += 1;

        ctx.decode(&mut batch)
            .with_context(|| "failed to decode generated token")?;
    }

    for outcome in classifier.flush() {
        totals.generated.push_str(&outcome.raw_piece);
        match outcome.sampled_token {
            SampledToken::Content(_) => totals.observed_content += 1,
            SampledToken::Reasoning(_) => totals.observed_reasoning += 1,
            SampledToken::ToolCall(_) | SampledToken::Undeterminable(_) => {}
        }
    }

    Ok(totals)
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
fn multimodal_vision_inference_produces_output(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    assert!(
        mtmd_ctx.support_vision(),
        "model should support vision input"
    );

    let image_path = test_model::fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)
        .with_context(|| "failed to load image from file")?;

    let formatted_prompt =
        build_user_prompt_with_image_marker(model, "What animals do you see in this image?")?;

    let input_text = MtmdInputText {
        text: formatted_prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let expected = count_chunk_tokens_by_type(&chunks)?;

    eprintln!(
        "tokenized into {} chunks, text {} image {} audio {}",
        chunks.len(),
        expected.text,
        expected.image,
        expected.audio
    );

    assert!(
        expected.image > 0,
        "vision input must produce at least one image chunk"
    );

    let mut classifier = model.sampled_token_classifier();
    let n_past = classifier
        .eval_multimodal_chunks(&chunks, mtmd_ctx, &ctx, 0, 0, 512, true)
        .with_context(|| "failed to evaluate chunks")?;

    eprintln!("evaluated chunks, n_past = {n_past}");

    {
        let usage = classifier.usage();
        assert_eq!(usage.prompt_tokens, expected.text);
        assert_eq!(usage.input_image_tokens, expected.image);
        assert_eq!(usage.input_audio_tokens, expected.audio);
    }

    let totals = drive_sampling_loop(&mut classifier, model, &mut ctx, n_past, 512)?;

    eprintln!("generated text: {}", totals.generated);

    assert!(
        !totals.generated.is_empty(),
        "model should generate at least one token from image input"
    );

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, expected.text);
    assert_eq!(usage.input_image_tokens, expected.image);
    assert_eq!(usage.input_audio_tokens, expected.audio);
    assert_eq!(usage.content_tokens, totals.observed_content);
    assert_eq!(usage.reasoning_tokens, totals.observed_reasoning);
    assert_eq!(
        usage.completion_tokens(),
        totals.observed_content + totals.observed_reasoning
    );

    Ok(())
}

llama_tests_main!();
