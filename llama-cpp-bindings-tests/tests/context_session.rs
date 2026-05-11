use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings_tests::FixtureSession;
use serial_test::serial;

#[test]
#[serial]
fn save_and_load_session_file() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_session.bin");
    context.state_save_file(&session_path, &tokens)?;

    let loaded_tokens = context.state_load_file(&session_path, 512)?;
    assert_eq!(loaded_tokens, tokens);

    std::fs::remove_file(&session_path)?;

    Ok(())
}

#[test]
#[serial]
fn get_state_size_is_positive() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    assert!(context.get_state_size() > 0);

    Ok(())
}

#[test]
#[serial]
fn state_seq_save_and_load_file_roundtrip() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_seq_state.bin");
    let bytes_written = context.state_seq_save_file(&session_path, 0, &tokens)?;
    assert!(bytes_written > 0);

    let (loaded_tokens, bytes_read) = context.state_seq_load_file(&session_path, 0, 512)?;
    assert_eq!(loaded_tokens, tokens);
    assert!(bytes_read > 0);

    std::fs::remove_file(&session_path)?;

    Ok(())
}

#[test]
#[serial]
fn copy_state_data_and_set_state_data_roundtrip() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let state_size = context.get_state_size();
    let mut state_data = vec![0u8; state_size];
    let bytes_copied = unsafe { context.copy_state_data(&mut state_data) };
    assert!(bytes_copied > 0);

    let bytes_read = unsafe { context.set_state_data(&state_data) };
    assert!(bytes_read > 0);

    Ok(())
}

#[test]
#[serial]
fn state_load_file_with_nonexistent_file_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.state_load_file("/nonexistent/session.bin", 512);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_seq_load_file_with_nonexistent_file_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let result = context.state_seq_load_file("/nonexistent/seq_state.bin", 0, 512);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_save_file_to_invalid_directory_returns_failed_to_save() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let result = context.state_save_file("/nonexistent_dir/session.bin", &[]);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_seq_save_file_to_invalid_directory_returns_failed_to_save() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let result = context.state_seq_save_file("/nonexistent_dir/seq_state.bin", 0, &[]);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_load_file_with_zero_max_tokens_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_session_zero_max.bin");
    context.state_save_file(&session_path, &tokens)?;

    let result = context.state_load_file(&session_path, 0);

    assert!(result.is_err());
    let _ = std::fs::remove_file(&session_path);

    Ok(())
}

#[test]
#[serial]
fn state_seq_load_file_with_zero_max_tokens_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_seq_state_zero_max.bin");
    context.state_seq_save_file(&session_path, 0, &tokens)?;

    let result = context.state_seq_load_file(&session_path, 0, 0);

    assert!(result.is_err());
    let _ = std::fs::remove_file(&session_path);

    Ok(())
}

#[test]
#[serial]
fn state_load_file_with_insufficient_max_tokens_returns_length_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token(
        "Hello world this is a longer string for more tokens",
        AddBos::Always,
    )?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_session_insuf.bin");
    context.state_save_file(&session_path, &tokens)?;

    let result = context.state_load_file(&session_path, 1);

    assert!(result.is_err());
    let _ = std::fs::remove_file(&session_path);

    Ok(())
}

#[test]
#[serial]
fn state_seq_load_file_with_insufficient_max_tokens_returns_length_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token(
        "Hello world this is a longer string for more tokens",
        AddBos::Always,
    )?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let session_path = std::env::temp_dir().join("llama_test_seq_state_insuf.bin");
    context.state_seq_save_file(&session_path, 0, &tokens)?;

    let result = context.state_seq_load_file(&session_path, 0, 1);

    assert!(result.is_err());
    let _ = std::fs::remove_file(&session_path);

    Ok(())
}

#[cfg(unix)]
#[test]
#[serial]
fn state_save_file_with_non_utf8_path_returns_error() -> Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
    let result = context.state_save_file(non_utf8_path, &[]);

    assert!(result.is_err());

    Ok(())
}

#[cfg(unix)]
#[test]
#[serial]
fn state_load_file_with_non_utf8_path_returns_error() -> Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
    let result = context.state_load_file(non_utf8_path, 512);

    assert!(result.is_err());

    Ok(())
}

#[cfg(unix)]
#[test]
#[serial]
fn state_seq_save_file_with_non_utf8_path_returns_error() -> Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
    let result = context.state_seq_save_file(non_utf8_path, 0, &[]);

    assert!(result.is_err());

    Ok(())
}

#[cfg(unix)]
#[test]
#[serial]
fn state_seq_load_file_with_non_utf8_path_returns_error() -> Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
    let result = context.state_seq_load_file(non_utf8_path, 0, 512);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_save_file_with_null_byte_in_path_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
    let result = context.state_save_file(path_with_null, &[]);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_load_file_with_null_byte_in_path_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
    let result = context.state_load_file(path_with_null, 512);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_seq_save_file_with_null_byte_in_path_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
    let result = context.state_seq_save_file(path_with_null, 0, &[]);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_seq_load_file_with_null_byte_in_path_returns_error() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
    let result = context.state_seq_load_file(path_with_null, 0, 512);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn state_seq_get_size_ext_returns_size_for_decoded_sequence() -> Result<()> {
    use llama_cpp_bindings::context::llama_state_seq_flags::LlamaStateSeqFlags;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let flags = LlamaStateSeqFlags::empty();
    let size = context.state_seq_get_size_ext(0, &flags);

    assert!(size > 0);

    Ok(())
}

#[test]
#[serial]
fn state_seq_get_data_ext_and_set_data_ext_round_trip() -> Result<()> {
    use llama_cpp_bindings::context::llama_state_seq_flags::LlamaStateSeqFlags;

    let fixture = FixtureSession::open()?;
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token("Hello world", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let flags = LlamaStateSeqFlags::empty();
    let size = context.state_seq_get_size_ext(0, &flags);
    let mut buffer = vec![0u8; size];
    let bytes_written = unsafe { context.state_seq_get_data_ext(&mut buffer, 0, &flags) };

    assert!(bytes_written > 0);

    let bytes_read = unsafe { context.state_seq_set_data_ext(&buffer, 0, &flags) };

    assert!(bytes_read > 0);

    Ok(())
}
