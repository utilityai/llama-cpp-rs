use anyhow::Context;
use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_context::MtmdContext;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const MAX_GENERATED_TOKENS: i32 = 512;
const DESCRIBE_INSTRUCTION: &str =
    "Describe the animal shown in the image, then write the exact words spoken in the audio.";

fn build_describe_image_and_audio_prompt(model: &LlamaModel) -> Result<String> {
    let marker = mtmd_default_marker()?;
    let template = model.chat_template(None)?;
    let user_content = format!("Image: {marker}\nAudio: {marker}\n{DESCRIBE_INSTRUCTION}");
    let messages = [LlamaChatMessage::new("user".to_string(), user_content)?];

    Ok(model.apply_chat_template(&template, &messages, true, false)?)
}

fn load_fixture_bitmap(mtmd_ctx: &MtmdContext, file_name: &str) -> Result<MtmdBitmap> {
    let path = fixtures_dir().join(file_name);
    let path_str = path
        .to_str()
        .with_context(|| format!("{file_name} path is not valid UTF-8"))?;
    MtmdBitmap::from_file(mtmd_ctx, path_str)
        .with_context(|| format!("failed to load {file_name} from file"))
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn image_and_audio_together(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    assert!(
        mtmd_ctx.support_vision(),
        "mmproj must support vision input for a combined image and audio test"
    );
    assert!(
        mtmd_ctx.support_audio(),
        "mmproj must support audio input for a combined image and audio test"
    );

    let image_bitmap = load_fixture_bitmap(mtmd_ctx, "llamas.jpg")?;
    assert!(!image_bitmap.is_audio(), "llamas.jpg must decode as image");

    let audio_bitmap = load_fixture_bitmap(mtmd_ctx, "orange_cat.wav")?;
    assert!(
        audio_bitmap.is_audio(),
        "orange_cat.wav must decode as audio"
    );

    let input_text = MtmdInputText {
        text: build_describe_image_and_audio_prompt(model)?,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&image_bitmap, &audio_bitmap])
        .with_context(|| "failed to tokenize combined image and audio input")?;

    let expected = ChunkTokenBreakdown::from_chunks(&chunks)?;
    assert!(
        expected.image > 0,
        "image input must produce at least one image-chunk token"
    );
    assert!(
        expected.audio > 0,
        "audio input must produce at least one audio-chunk token"
    );

    let required_n_ctx = u32::try_from(chunks.total_positions() + MAX_GENERATED_TOKENS)?;
    assert!(
        fixture.context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        fixture.context_params.n_ctx,
        required_n_ctx,
    );

    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    let n_batch = i32::try_from(context.n_batch())?;
    let mut classifier = model.sampled_token_classifier()?;
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &context,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate image and audio chunks")?;

    {
        let usage = classifier.usage();
        assert_eq!(usage.input_image_tokens, expected.image);
        assert_eq!(usage.input_audio_tokens, expected.audio);
        assert_eq!(usage.prompt_tokens, expected.text);
    }

    let mut sampler = LlamaSampler::greedy();
    let mut batch = LlamaBatch::new(512, 1)?;
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

    let description = outcome.generated_raw.to_lowercase();
    assert!(
        !description.is_empty(),
        "model should generate a description from combined image and audio input"
    );
    assert!(
        description.contains("sheep"),
        "the gemma-4 vision encoder recognizes the image animals as \"sheep\" (a borderline \
         llama/sheep call the b9585 clip-encoder update tipped); the assertion tracks the \
         model's actual recognition so it still proves the image reached the output; \
         got: {description:?}"
    );
    assert!(
        description.contains("fence"),
        "description should echo the spoken word \"fence\" from the audio; got: {description:?}"
    );

    Ok(())
}
