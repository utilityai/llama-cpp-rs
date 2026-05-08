use anyhow::Result;
use llama_cpp_bindings::ingest_prompt_chunk::ingest_prompt_chunk;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings_tests::TestFixture;
use llama_cpp_bindings_tests::test_model::fixtures_dir;

#[test]
fn text_chunk_records_prompt_tokens() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let input_text = MtmdInputText {
        text: "hello world".to_owned(),
        add_special: false,
        parse_special: false,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[])?;

    let text_chunk = (0..chunks.len())
        .filter_map(|index| chunks.get(index))
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Text))
        .ok_or_else(|| anyhow::anyhow!("text-only tokenization should produce at least one text chunk"))?;

    let n_tokens = text_chunk.n_tokens() as u64;

    let mut classifier = model.sampled_token_classifier();

    ingest_prompt_chunk(&mut classifier, &text_chunk)?;

    let usage = classifier.usage();
    if usage.prompt_tokens != n_tokens {
        anyhow::bail!(
            "text chunk must record n_tokens as prompt_tokens; expected {n_tokens}, got {}",
            usage.prompt_tokens
        );
    }
    if usage.input_image_tokens != 0 {
        anyhow::bail!(
            "text chunk must not bump input_image_tokens; got {}",
            usage.input_image_tokens
        );
    }
    if usage.input_audio_tokens != 0 {
        anyhow::bail!(
            "text chunk must not bump input_audio_tokens; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}

#[test]
fn image_chunk_records_input_image_tokens_only() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker();
    let input_text = MtmdInputText {
        text: marker.to_owned(),
        add_special: false,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let image_chunk = (0..chunks.len())
        .filter_map(|index| chunks.get(index))
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Image))
        .ok_or_else(|| anyhow::anyhow!("multimodal tokenization should produce an image chunk"))?;

    let n_tokens = image_chunk.n_tokens() as u64;
    if n_tokens == 0 {
        anyhow::bail!("image chunk should report at least one token");
    }

    let mut classifier = model.sampled_token_classifier();

    ingest_prompt_chunk(&mut classifier, &image_chunk)?;

    let usage = classifier.usage();
    if usage.input_image_tokens != n_tokens {
        anyhow::bail!(
            "image chunk must record n_tokens as input_image_tokens; expected {n_tokens}, got {}",
            usage.input_image_tokens
        );
    }
    if usage.prompt_tokens != 0 {
        anyhow::bail!(
            "image chunk must not bump prompt_tokens; got {}",
            usage.prompt_tokens
        );
    }
    if usage.input_audio_tokens != 0 {
        anyhow::bail!(
            "image chunk must not bump input_audio_tokens; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}

#[test]
fn text_chunk_drives_marker_state_machine_to_reasoning() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let input_text = MtmdInputText {
        text: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n".to_owned(),
        add_special: false,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[])?;

    let mut classifier = model.sampled_token_classifier();

    for index in 0..chunks.len() {
        let chunk = chunks
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("chunk index {index} must exist"))?;
        ingest_prompt_chunk(&mut classifier, &chunk)?;
    }

    if classifier.current_section() != llama_cpp_bindings::SampledTokenSection::Reasoning {
        anyhow::bail!(
            "text chunk replay must transition the classifier section to Reasoning when the \
             prompt opens a `<think>` block; got {:?}",
            classifier.current_section()
        );
    }

    Ok(())
}
