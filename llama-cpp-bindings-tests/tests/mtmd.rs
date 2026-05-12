#![cfg(feature = "multimodal_capable")]

use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;
use llama_cpp_bindings::mtmd::MtmdEvalError;
use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputChunks;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings_tests::FixtureSession;
use llama_cpp_bindings_tests::test_model;
use serial_test::serial;

fn eval_synthetic_bitmap(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    width: u32,
    height: u32,
) -> Result<()> {
    let image_data = vec![128u8; (width as usize) * (height as usize) * 3];
    let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let context_size = u32::try_from(n_positions + 256).unwrap_or(8192);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(context_size));
    let llama_ctx = LlamaContext::from_model(model, backend, ctx_params)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, false)?;

    Ok(())
}

#[test]
#[serial]
fn eval_chunks_returns_batch_size_exceeds_context_limit_for_huge_batch() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(64));
    let llama_ctx = LlamaContext::from_model(model, backend, ctx_params)?;

    let chunks = MtmdInputChunks::new()?;
    let huge_batch = i32::try_from(llama_ctx.n_batch() + 1)?;

    let result = chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, huge_batch, false);

    assert!(matches!(
        result,
        Err(MtmdEvalError::BatchSizeExceedsContextLimit { .. })
    ));

    Ok(())
}

#[test]
#[serial]
fn from_buffer_creates_bitmap_from_image_bytes() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let fixtures = test_model::fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_bytes = std::fs::read(&image_path)?;
    let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, &image_bytes)?;

    assert!(bitmap.nx() > 0);
    assert!(bitmap.ny() > 0);
    assert!(!bitmap.is_audio());

    Ok(())
}

#[test]
#[serial]
fn from_file_with_null_byte_in_path_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;
    let result = MtmdBitmap::from_file(mtmd_ctx, "path\0null");

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn text_chunk_has_text_type() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello world <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;

    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);

    Ok(())
}

#[test]
#[serial]
fn text_chunk_returns_text_tokens() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello world <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    let tokens = first_chunk.text_tokens();

    assert!(tokens.is_some());
    assert!(!tokens.expect("tokens should be some").is_empty());

    Ok(())
}

#[test]
#[serial]
fn chunk_n_tokens_is_positive() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello world <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;

    assert!(first_chunk.n_tokens() > 0);

    Ok(())
}

#[test]
#[serial]
fn chunk_n_positions_is_positive() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello world <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;

    assert!(first_chunk.n_positions() > 0);

    Ok(())
}

#[test]
#[serial]
fn copy_creates_owned_duplicate() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    let copied = first_chunk.copy()?;

    assert!(copied.owned);
    assert_eq!(copied.n_tokens(), first_chunk.n_tokens());

    Ok(())
}

#[test]
#[serial]
fn text_chunk_id_returns_none() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;

    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);
    assert!(first_chunk.id().is_none());

    Ok(())
}

#[test]
#[serial]
fn image_chunk_returns_none_for_text_tokens() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .ok_or_else(|| anyhow::anyhow!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
            assert!(chunk.text_tokens().is_none());

            return Ok(());
        }
    }

    Ok(())
}

#[test]
#[serial]
fn image_chunk_id_returns_some() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .ok_or_else(|| anyhow::anyhow!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
            assert!(chunk.id().is_some());

            return Ok(());
        }
    }

    Ok(())
}

#[test]
#[serial]
fn init_and_supports_vision() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    assert!(mtmd_ctx.support_vision());

    Ok(())
}

#[test]
#[serial]
fn tokenize_text_with_image() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe this image: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    assert!(!chunks.is_empty());
    assert!(chunks.total_tokens() > 0);

    Ok(())
}

#[test]
#[serial]
fn eval_chunks_with_standard_image() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let fixtures = test_model::fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;
    let input_text = MtmdInputText {
        text: "What is in this image? <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let context_size = u32::try_from(n_positions + 256).unwrap_or(2048);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(context_size));
    let llama_ctx = LlamaContext::from_model(model, backend, ctx_params)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let result = chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, false);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn eval_chunks_with_varied_dimensions() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let test_dimensions: [(u32, u32); 4] = [(224, 224), (512, 512), (100, 500), (337, 421)];

    for (width, height) in test_dimensions {
        let result = eval_synthetic_bitmap(backend, model, mtmd_ctx, width, height);

        assert!(
            result.is_ok(),
            "dimension {width}x{height} should succeed: {result:?}"
        );
    }

    Ok(())
}

#[test]
#[serial]
fn decode_use_non_causal_returns_bool() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Hello world <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    let _non_causal = mtmd_ctx.decode_use_non_causal(&first_chunk);

    Ok(())
}

#[test]
#[serial]
fn decode_use_mrope_returns_bool() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let _mrope = mtmd_ctx.decode_use_mrope();

    Ok(())
}

#[test]
#[serial]
fn support_audio_returns_bool() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let _audio = mtmd_ctx.support_audio();

    Ok(())
}

#[test]
#[serial]
fn get_audio_sample_rate_returns_option() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let _rate = mtmd_ctx.get_audio_sample_rate();

    Ok(())
}

#[test]
#[serial]
fn encode_chunk_succeeds_for_image_chunk() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .ok_or_else(|| anyhow::anyhow!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
            let result = mtmd_ctx.encode_chunk(&chunk);

            assert!(result.is_ok());

            return Ok(());
        }
    }

    Ok(())
}

#[test]
#[serial]
fn tokenize_bitmap_count_mismatch_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let input_text = MtmdInputText {
        text: "No media markers here".to_string(),
        add_special: true,
        parse_special: true,
    };
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let result = mtmd_ctx.tokenize(input_text, &[&bitmap]);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn eval_chunks_with_extreme_dimensions_does_not_crash() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let mtmd_ctx = fixture.mtmd_context()?;

    let extreme_dimensions: [(u32, u32); 6] = [
        (1, 1),
        (7, 13),
        (3, 1000),
        (1000, 3),
        (1920, 1080),
        (4096, 4096),
    ];

    let mut any_reached_eval = false;

    for (width, height) in extreme_dimensions {
        match eval_synthetic_bitmap(backend, model, mtmd_ctx, width, height) {
            Ok(()) => any_reached_eval = true,
            Err(error) => eprintln!("  {width}x{height} failed: {error}"),
        }
    }

    assert!(
        any_reached_eval,
        "at least one extreme dimension should reach eval_chunks"
    );

    Ok(())
}

#[test]
#[serial]
fn init_from_file_with_null_byte_in_path_returns_error() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();
    let mtmd_params = MtmdContextParams::default();
    let result = MtmdContext::init_from_file("path\0null", model, &mtmd_params);

    assert!(result.is_err());
}

#[test]
#[serial]
fn tokenize_with_null_byte_in_text_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let mtmd_ctx = fixture.mtmd_context()?;

    let input_text = MtmdInputText {
        text: "text\0null".to_string(),
        add_special: true,
        parse_special: true,
    };
    let result = mtmd_ctx.tokenize(input_text, &[]);

    assert!(result.is_err());

    Ok(())
}
