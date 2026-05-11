use std::num::NonZeroU8;
use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::kv_cache::KvCacheConversionError;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings_tests::FixtureSession;
use serial_test::serial;

#[test]
#[serial]
fn clear_kv_cache_resets_positions() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    context.clear_kv_cache();
    assert_eq!(context.kv_cache_seq_pos_max(0), -1);

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_pos_max_is_non_negative_after_decode() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    assert!(context.kv_cache_seq_pos_max(0) >= 0);

    Ok(())
}

#[test]
#[serial]
fn clear_kv_cache_seq_with_range() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.clear_kv_cache_seq(Some(0), Some(0), Some(1));
    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn copy_kv_cache_seq_succeeds() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.copy_kv_cache_seq(0, 1, None, None);
    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn copy_cache_executes_without_crash() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let pos_max = context.kv_cache_seq_pos_max(0);
    context.copy_cache(0, 1, pos_max + 1);

    Ok(())
}

#[cfg(feature = "mrope_model")]
#[test]
#[serial]
fn kv_cache_seq_add_returns_error_for_mrope_model() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.kv_cache_seq_add(0, Some(0), None, 1);

    assert!(result.is_err());

    Ok(())
}

#[cfg(feature = "mrope_model")]
#[test]
#[serial]
fn kv_cache_seq_div_returns_error_for_mrope_model() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let divisor = NonZeroU8::new(2).ok_or_else(|| anyhow::anyhow!("2 is non-zero"))?;
    let result = context.kv_cache_seq_div(0, Some(0), None, divisor);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_keep_retains_specified_sequence() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    context.kv_cache_seq_keep(0);

    assert!(context.kv_cache_seq_pos_max(0) >= 0);

    Ok(())
}

#[test]
#[serial]
fn copy_kv_cache_seq_with_explicit_range() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.copy_kv_cache_seq(0, 2, Some(0), Some(1));

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_add_succeeds_on_embedding_model() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.kv_cache_seq_add(0, Some(0), None, 1);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_div_succeeds_on_embedding_model() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let divisor = NonZeroU8::new(2).ok_or_else(|| anyhow::anyhow!("2 is non-zero"))?;
    let result = context.kv_cache_seq_div(0, Some(0), None, divisor);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_pos_max_returns_negative_one_for_unused_seq() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let result = context.kv_cache_seq_pos_max(999);

    assert_eq!(result, -1);

    Ok(())
}

#[test]
#[serial]
fn copy_kv_cache_seq_rejects_p0_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.copy_kv_cache_seq(0, 1, Some(u32::MAX), None);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P0TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn copy_kv_cache_seq_rejects_p1_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.copy_kv_cache_seq(0, 1, Some(0), Some(u32::MAX));

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P1TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn clear_kv_cache_seq_rejects_src_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.clear_kv_cache_seq(Some(u32::MAX), None, None);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::SeqIdTooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn clear_kv_cache_seq_rejects_p0_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.clear_kv_cache_seq(Some(0), Some(u32::MAX), None);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P0TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn clear_kv_cache_seq_rejects_p1_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.clear_kv_cache_seq(Some(0), Some(0), Some(u32::MAX));

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P1TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_add_rejects_p0_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.kv_cache_seq_add(0, Some(u32::MAX), None, 1);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P0TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_add_rejects_p1_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.kv_cache_seq_add(0, Some(0), Some(u32::MAX), 1);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P1TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_div_rejects_p0_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let divisor = NonZeroU8::new(2).ok_or_else(|| anyhow::anyhow!("2 is non-zero"))?;
    let result = context.kv_cache_seq_div(0, Some(u32::MAX), None, divisor);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P0TooLarge(_),
    ));

    Ok(())
}

#[test]
#[serial]
fn kv_cache_seq_div_rejects_p1_exceeding_i32_max() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let divisor = NonZeroU8::new(2).ok_or_else(|| anyhow::anyhow!("2 is non-zero"))?;
    let result = context.kv_cache_seq_div(0, Some(0), Some(u32::MAX), divisor);

    assert!(matches!(
        result.unwrap_err(),
        KvCacheConversionError::P1TooLarge(_),
    ));

    Ok(())
}
