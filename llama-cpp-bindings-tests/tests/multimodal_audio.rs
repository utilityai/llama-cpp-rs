use anyhow::Context;
use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const TRANSCRIBE_SYSTEM_PROMPT: &str =
    "You are a helpful assistant that can hear audio and write down the words that are spoken.";
const TRANSCRIBE_INSTRUCTION: &str = "What words are spoken in this audio?";

fn assert_audio_transcription_contains(
    fixture: &LlamaFixture<'_>,
    audio_file_name: &str,
    expected_word: &str,
) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    assert!(
        mtmd_ctx.support_audio(),
        "mmproj must support audio input for an audio transcription test"
    );

    let marker = mtmd_default_marker()?;
    let template = model.chat_template(None)?;
    let messages = [
        LlamaChatMessage::new("system".to_string(), TRANSCRIBE_SYSTEM_PROMPT.to_string())?,
        LlamaChatMessage::new(
            "user".to_string(),
            format!("{marker}{TRANSCRIBE_INSTRUCTION}"),
        )?,
    ];
    let input_text = MtmdInputText {
        text: model.apply_chat_template(&template, &messages, true, true)?,
        add_special: false,
        parse_special: true,
    };

    let audio_path = fixtures_dir().join(audio_file_name);
    let audio_path_str = audio_path
        .to_str()
        .with_context(|| "audio path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, audio_path_str)
        .with_context(|| "failed to load audio from file")?;

    assert!(bitmap.is_audio(), "fixture must decode as audio");

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal audio input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let mut classifier = model.sampled_token_classifier()?;
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &context,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 512,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate audio chunks")?;

    {
        let usage = classifier.usage();
        assert!(
            usage.input_audio_tokens > 0,
            "audio input must record audio prompt tokens; got {}",
            usage.input_audio_tokens
        );
        assert_eq!(
            usage.input_image_tokens, 0,
            "audio-only input must not record image tokens"
        );
        assert!(
            usage.prompt_tokens > 0,
            "the text portion of the prompt must record prompt tokens"
        );
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
        max_generated_tokens: 512,
    }
    .run()?;

    let transcript = outcome.generated_raw.to_lowercase();
    assert!(
        !transcript.is_empty(),
        "model should generate content from audio input"
    );
    assert!(
        transcript.contains(expected_word),
        "transcription should echo the spoken word {expected_word:?}; got: {transcript:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
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
fn audio_mmproj_reports_audio_support(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    assert!(
        mtmd_ctx.support_audio(),
        "an audio mmproj must report audio support"
    );
    assert!(
        mtmd_ctx.get_audio_sample_rate().is_some(),
        "an audio-capable mmproj must report a required sample rate"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
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
fn audio_transcribes_spoken_word(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_audio_transcription_contains(fixture, "quick_brown_fox.wav", "fox")
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
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
fn audio_transcribes_uncommon_sentence(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_audio_transcription_contains(fixture, "orange_cat.wav", "fence")
}
