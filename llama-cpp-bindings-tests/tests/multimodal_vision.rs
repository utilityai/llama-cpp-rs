#![expect(
    clippy::unnecessary_wraps,
    reason = "trial fns share the harness LlamaTestFn signature even when their bodies never propagate"
)]

use anyhow::Context;
use anyhow::Result;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::TokenUsage;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::ingest_prompt_chunk::ingest_prompt_chunk;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;
use llama_cpp_bindings::mtmd::MtmdEvalError;
use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputChunks;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_tests::build_user_prompt_with_media_marker::build_user_prompt_with_media_marker;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn from_buffer_creates_bitmap_from_image_bytes(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let fixtures = fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_bytes = std::fs::read(&image_path)?;
    let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, &image_bytes)?;

    assert!(bitmap.nx() > 0);
    assert!(bitmap.ny() > 0);
    assert!(!bitmap.is_audio());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn from_file_with_null_byte_in_path_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let result = MtmdBitmap::from_file(mtmd_ctx, "path\0null");

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn copy_creates_owned_duplicate(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn encode_chunk_succeeds_for_image_chunk(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn decode_use_non_causal_returns_bool_for_image_chunk(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
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
            let value = mtmd_ctx.decode_use_non_causal(&chunk);
            let printed = format!("{value:?}");
            assert!(
                !printed.is_empty(),
                "decode_use_non_causal must return a Debug-printable bool"
            );
            return Ok(());
        }
    }
    anyhow::bail!("tokenization should produce at least one Image chunk");
}

fn tokenize_synthetic(fixture: &LlamaFixture<'_>, prompt: &str) -> Result<MtmdInputChunks> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: prompt.to_owned(),
        add_special: true,
        parse_special: true,
    };
    Ok(mtmd_ctx.tokenize(input_text, &[&bitmap])?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_has_text_type(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello world <__media__>")?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_returns_text_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello world <__media__>")?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    let tokens = first_chunk.text_tokens();
    assert!(tokens.is_some());
    assert!(!tokens.expect("tokens should be some").is_empty());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_n_tokens_is_positive(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello world <__media__>")?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    assert!(first_chunk.n_tokens() > 0);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_n_positions_is_positive(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello world <__media__>")?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    assert!(first_chunk.n_positions() > 0);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_id_returns_none(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello <__media__>")?;
    let first_chunk = chunks
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("missing first chunk"))?;
    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);
    assert!(first_chunk.id().is_none());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_returns_none_for_text_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello <__media__>")?;
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_id_returns_some(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_synthetic(fixture, "Hello <__media__>")?;
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_and_supports_vision(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(mtmd_ctx.support_vision());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn init_from_file_with_null_byte_in_path_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_params = MtmdContextParams::default();
    let result = MtmdContext::init_from_file("path\0null", fixture.model, &mtmd_params);

    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn decode_use_mrope_is_true_for_qwen_vision(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        mtmd_ctx.decode_use_mrope(),
        "Qwen 3.5 / 3.6 mmproj uses mrope; decode_use_mrope must return true"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn support_audio_is_false_for_vision_only_mmproj(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        !mtmd_ctx.support_audio(),
        "Qwen 3.5 / 3.6 mmproj is vision-only; support_audio must return false"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn get_audio_sample_rate_is_none_for_vision_only_mmproj(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    assert!(
        mtmd_ctx.get_audio_sample_rate().is_none(),
        "Qwen 3.5 / 3.6 mmproj has no audio; get_audio_sample_rate must return None"
    );
    Ok(())
}

fn eval_synthetic_bitmap(fixture: &LlamaFixture<'_>, width: u32, height: u32) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; (width as usize) * (height as usize) * 3];
    let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(n_positions + 256)?;
    if fixture.context_params.n_ctx < required_n_ctx {
        anyhow::bail!(
            "fixture n_ctx ({}) below required ({}) for {}x{} image",
            fixture.context_params.n_ctx,
            required_n_ctx,
            width,
            height,
        );
    }

    let llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, false)?;
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_returns_batch_size_exceeds_context_limit_for_huge_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chunks = MtmdInputChunks::new()?;
    let huge_batch = i32::try_from(llama_ctx.n_batch() + 1)?;

    let result = chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, huge_batch, false);

    assert!(matches!(
        result,
        Err(MtmdEvalError::BatchSizeExceedsContextLimit { .. })
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_with_standard_image(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let fixtures = fixtures_dir();
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
    let required_n_ctx = u32::try_from(n_positions + 256)?;
    assert!(
        fixture.context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        fixture.context_params.n_ctx,
        required_n_ctx,
    );

    let llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let result = chunks.eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, false);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_with_varied_dimensions(fixture: &LlamaFixture<'_>) -> Result<()> {
    let test_dimensions: [(u32, u32); 4] = [(224, 224), (512, 512), (100, 500), (337, 421)];

    for (width, height) in test_dimensions {
        let result = eval_synthetic_bitmap(fixture, width, height);
        assert!(
            result.is_ok(),
            "dimension {width}x{height} should succeed: {result:?}"
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_with_extreme_dimensions_does_not_crash(fixture: &LlamaFixture<'_>) -> Result<()> {
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
        match eval_synthetic_bitmap(fixture, width, height) {
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenize_text_with_image(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenize_bitmap_count_mismatch_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenize_with_null_byte_in_text_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let input_text = MtmdInputText {
        text: "text\0null".to_string(),
        add_special: true,
        parse_special: true,
    };
    let result = mtmd_ctx.tokenize(input_text, &[]);
    assert!(result.is_err());
    Ok(())
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

    for (current_position, _) in (starting_position..).zip(0..max_tokens) {
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

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)
        .with_context(|| "failed to load image from file")?;

    let formatted_prompt =
        build_user_prompt_with_media_marker(model, "What animals do you see in this image?")?;

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

    let expected = ChunkTokenBreakdown::from_chunks(&chunks)?;

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

    let mut classifier = model.sampled_token_classifier()?;
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

const PROMPT_QUESTION: &str = "What animals do you see in this image?";

fn build_multimodal_chunks_and_eval_into_usage(
    fixture: &LlamaFixture<'_>,
) -> Result<(TokenUsage, ChunkTokenBreakdown)> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker()?;
    let prompt = format!("{marker}{PROMPT_QUESTION}");

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let expected = ChunkTokenBreakdown::from_chunks(&chunks)?;

    let context_params = (*fixture.context_params).into_llama_context_params();
    let context = LlamaContext::from_model(model, fixture.backend, context_params)?;

    let mut classifier = model.sampled_token_classifier()?;
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_records_prompt_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let input_text = MtmdInputText {
        text: "hello world".to_owned(),
        add_special: false,
        parse_special: false,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[])?;

    let text_chunk = (0..chunks.len())
        .filter_map(|index| chunks.get(index))
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Text))
        .ok_or_else(|| {
            anyhow::anyhow!("text-only tokenization should produce at least one text chunk")
        })?;

    let n_tokens = u64::try_from(text_chunk.n_tokens())?;

    let mut classifier = model.sampled_token_classifier()?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_records_input_image_tokens_only(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("image path is not valid UTF-8"))?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker()?;
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

    let n_tokens = u64::try_from(image_chunk.n_tokens())?;
    if n_tokens == 0 {
        anyhow::bail!("image chunk should report at least one token");
    }

    let mut classifier = model.sampled_token_classifier()?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_drives_marker_state_machine_to_reasoning(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let input_text = MtmdInputText {
        text: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n".to_owned(),
        add_special: false,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[])?;

    let mut classifier = model.sampled_token_classifier()?;

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
    const MAX_GENERATED_TOKENS: i32 = 200;

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

    let marker = mtmd_default_marker()?;
    let prompt = format!(
        "<bos><start_of_turn>user\n{marker}What animals do you see in this image?<end_of_turn>\n<start_of_turn>model\n<|channel>thought\n"
    );

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier = model.sampled_token_classifier()?;
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

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "mmproj-F16.gguf"),
)]
fn mistral3_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 512;

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

    let marker = mtmd_default_marker()?;
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

    let mut classifier = model.sampled_token_classifier()?;
    let n_past = classifier.eval_multimodal_chunks(&chunks, mtmd_ctx, &context, 0, 0, 512, true)?;

    let mut sampler = LlamaSampler::greedy();
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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn qwen35_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

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

    let marker = mtmd_default_marker()?;
    let prompt = format!(
        "<|im_start|>user\n{marker}What animals do you see in this image?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier = model.sampled_token_classifier()?;
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
            "Qwen 3.5 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the prompt opens a `<think>` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        anyhow::bail!(
            "Qwen 3.5 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn qwen36_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

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

    let marker = mtmd_default_marker()?;
    let prompt = format!(
        "<|im_start|>user\n{marker}What animals do you see in this image?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier = model.sampled_token_classifier()?;
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
            "Qwen 3.6 multimodal + thinking: classifier must emit at least one Reasoning token; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        anyhow::bail!(
            "Qwen 3.6 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}
