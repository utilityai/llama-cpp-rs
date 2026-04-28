//! utilities for working with session files

use crate::context::LlamaContext;
use crate::context::llama_state_seq_flags::LlamaStateSeqFlags;
use crate::context::load_seq_state_error::LoadSeqStateError;
use crate::context::load_session_error::LoadSessionError;
use crate::context::save_seq_state_error::SaveSeqStateError;
use crate::context::save_session_error::SaveSessionError;
use crate::token::LlamaToken;
use std::ffi::CString;
use std::path::Path;

fn process_session_load_result(
    success: bool,
    n_out: usize,
    max_tokens: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<Vec<LlamaToken>, LoadSessionError> {
    if !success {
        return Err(LoadSessionError::FailedToLoad);
    }

    if n_out > max_tokens {
        return Err(LoadSessionError::InsufficientMaxLength { n_out, max_tokens });
    }

    unsafe { tokens.set_len(n_out) };

    Ok(tokens)
}

fn process_seq_load_result(
    bytes_read: usize,
    n_out: usize,
    max_tokens: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<(Vec<LlamaToken>, usize), LoadSeqStateError> {
    if bytes_read == 0 {
        return Err(LoadSeqStateError::FailedToLoad);
    }

    if n_out > max_tokens {
        return Err(LoadSeqStateError::InsufficientMaxLength { n_out, max_tokens });
    }

    unsafe { tokens.set_len(n_out) };

    Ok((tokens, bytes_read))
}

impl LlamaContext<'_> {
    /// Save the full state to a file.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to save to.
    /// * `tokens` - The tokens to associate the state with. This should be a prefix of a sequence
    ///   of tokens that the context has processed, so that the relevant KV caches are already filled.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the state file.
    pub fn state_save_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), SaveSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        if unsafe {
            llama_cpp_bindings_sys::llama_state_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }

    /// Load a state file into the current context.
    ///
    /// You still need to pass the returned tokens to the context for inference to work. What this
    /// function buys you is that the KV caches are already filled with the relevant data.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to load from. It must be a state file from a compatible context,
    ///   otherwise the function will error.
    /// * `max_tokens` - The maximum token length of the loaded state. If the state was saved with a
    ///   longer length, the function will error.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the state file.
    pub fn state_load_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, LoadSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| LoadSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>();

        let success = unsafe {
            llama_cpp_bindings_sys::llama_state_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };
        process_session_load_result(success, n_out, max_tokens, tokens)
    }

    /// Save state for a single sequence to a file.
    ///
    /// This enables saving state for individual sequences, which is useful for multi-sequence
    /// inference scenarios.
    ///
    /// # Parameters
    ///
    /// * `filepath` - The file to save to.
    /// * `seq_id` - The sequence ID whose state to save.
    /// * `tokens` - The tokens to associate with the saved state.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the sequence state file.
    ///
    /// # Returns
    ///
    /// The number of bytes written on success.
    pub fn state_seq_save_file(
        &self,
        filepath: impl AsRef<Path>,
        seq_id: i32,
        tokens: &[LlamaToken],
    ) -> Result<usize, SaveSeqStateError> {
        let path = filepath.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSeqStateError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        let bytes_written = unsafe {
            llama_cpp_bindings_sys::llama_state_seq_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                seq_id,
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                tokens.len(),
            )
        };

        if bytes_written == 0 {
            Err(SaveSeqStateError::FailedToSave)
        } else {
            Ok(bytes_written)
        }
    }

    /// Load state for a single sequence from a file.
    ///
    /// This enables loading state for individual sequences, which is useful for multi-sequence
    /// inference scenarios.
    ///
    /// # Parameters
    ///
    /// * `filepath` - The file to load from.
    /// * `dest_seq_id` - The destination sequence ID to load the state into.
    /// * `max_tokens` - The maximum number of tokens to read.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the sequence state file.
    ///
    /// # Returns
    ///
    /// A tuple of `(tokens, bytes_read)` on success.
    pub fn state_seq_load_file(
        &mut self,
        filepath: impl AsRef<Path>,
        dest_seq_id: i32,
        max_tokens: usize,
    ) -> Result<(Vec<LlamaToken>, usize), LoadSeqStateError> {
        let path = filepath.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| LoadSeqStateError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>();

        let bytes_read = unsafe {
            llama_cpp_bindings_sys::llama_state_seq_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                dest_seq_id,
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };

        process_seq_load_result(bytes_read, n_out, max_tokens, tokens)
    }

    /// Returns the maximum size in bytes of the state (rng, logits, embedding
    /// and `kv_cache`) - will often be smaller after compacting tokens
    #[must_use]
    pub fn get_state_size(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::llama_state_get_size(self.context.as_ptr()) }
    }

    /// Copies the state to the specified destination buffer.
    ///
    /// Use [`get_state_size`](Self::get_state_size) to determine the required buffer size.
    ///
    /// Returns the number of bytes copied.
    ///
    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    pub unsafe fn copy_state_data(&self, dest: &mut [u8]) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
            )
        }
    }

    /// Set the state reading from the specified buffer.
    ///
    /// Returns the number of bytes read.
    ///
    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from [`copy_state_data`](Self::copy_state_data)
    /// on a compatible context (same model and parameters). Passing arbitrary or corrupted bytes
    /// will lead to undefined behavior.
    pub unsafe fn set_state_data(&mut self, src: &[u8]) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
            )
        }
    }

    /// Get the size of the state data for a specific sequence, with extended flags.
    ///
    /// Useful for hybrid/recurrent models where partial state (e.g., only SSM state)
    /// may be saved or restored.
    #[must_use]
    pub fn state_seq_get_size_ext(&self, seq_id: i32, flags: &LlamaStateSeqFlags) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_seq_get_size_ext(
                self.context.as_ptr(),
                seq_id,
                flags.bits(),
            )
        }
    }

    /// Copy state data for a specific sequence into `dest`, with extended flags.
    ///
    /// Use [`state_seq_get_size_ext`](Self::state_seq_get_size_ext) to determine the required
    /// buffer size before calling this method.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    pub unsafe fn state_seq_get_data_ext(
        &self,
        dest: &mut [u8],
        seq_id: i32,
        flags: &LlamaStateSeqFlags,
    ) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_seq_get_data_ext(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
                seq_id,
                flags.bits(),
            )
        }
    }

    /// Restore state data for a specific sequence from `src`, with extended flags.
    ///
    /// Returns the number of bytes read.
    ///
    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from
    /// [`state_seq_get_data_ext`](Self::state_seq_get_data_ext) on a compatible context.
    pub unsafe fn state_seq_set_data_ext(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
        flags: &LlamaStateSeqFlags,
    ) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_seq_set_data_ext(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
                flags.bits(),
            )
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::token::LlamaToken;

    use crate::context::load_seq_state_error::LoadSeqStateError;
    use crate::context::load_session_error::LoadSessionError;

    use super::{process_seq_load_result, process_session_load_result};

    #[test]
    fn session_load_success_within_bounds() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(true, 10, 100, tokens);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn session_load_fails_when_not_successful() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(false, 0, 100, tokens);

        assert_eq!(result, Err(LoadSessionError::FailedToLoad));
    }

    #[test]
    fn session_load_fails_when_n_out_exceeds_max() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(true, 101, 100, tokens);

        assert_eq!(
            result,
            Err(LoadSessionError::InsufficientMaxLength {
                n_out: 101,
                max_tokens: 100,
            })
        );
    }

    #[test]
    fn seq_load_success_within_bounds() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(42, 10, 100, tokens);

        assert!(result.is_ok());
        let (loaded, bytes) = result.unwrap();
        assert_eq!(loaded.len(), 10);
        assert_eq!(bytes, 42);
    }

    #[test]
    fn seq_load_fails_when_zero_bytes_read() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(0, 0, 100, tokens);

        assert_eq!(result, Err(LoadSeqStateError::FailedToLoad));
    }

    #[test]
    fn seq_load_fails_when_n_out_exceeds_max() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(42, 101, 100, tokens);

        assert_eq!(
            result,
            Err(LoadSeqStateError::InsufficientMaxLength {
                n_out: 101,
                max_tokens: 100,
            })
        );
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use std::num::NonZeroU32;

    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::llama_batch::LlamaBatch;
    use crate::model::AddBos;
    use crate::test_model;

    #[test]
    #[serial]
    fn save_and_load_session_file() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_session.bin");
        context.state_save_file(&session_path, &tokens).unwrap();

        let loaded_tokens = context.state_load_file(&session_path, 512).unwrap();
        assert_eq!(loaded_tokens, tokens);

        std::fs::remove_file(&session_path).unwrap();
    }

    #[test]
    #[serial]
    fn get_state_size_is_positive() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        assert!(context.get_state_size() > 0);
    }

    #[test]
    #[serial]
    fn state_seq_save_and_load_file_roundtrip() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_seq_state.bin");
        let bytes_written = context
            .state_seq_save_file(&session_path, 0, &tokens)
            .unwrap();
        assert!(bytes_written > 0);

        let (loaded_tokens, bytes_read) =
            context.state_seq_load_file(&session_path, 0, 512).unwrap();
        assert_eq!(loaded_tokens, tokens);
        assert!(bytes_read > 0);

        std::fs::remove_file(&session_path).unwrap();
    }

    #[test]
    #[serial]
    fn copy_state_data_and_set_state_data_roundtrip() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let state_size = context.get_state_size();
        let mut state_data = vec![0u8; state_size];
        let bytes_copied = unsafe { context.copy_state_data(&mut state_data) };
        assert!(bytes_copied > 0);

        let bytes_read = unsafe { context.set_state_data(&state_data) };
        assert!(bytes_read > 0);
    }

    #[test]
    #[serial]
    fn state_load_file_with_nonexistent_file_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.state_load_file("/nonexistent/session.bin", 512);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_seq_load_file_with_nonexistent_file_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.state_seq_load_file("/nonexistent/seq_state.bin", 0, 512);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_save_file_to_invalid_directory_returns_failed_to_save() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.state_save_file("/nonexistent_dir/session.bin", &[]);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_seq_save_file_to_invalid_directory_returns_failed_to_save() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.state_seq_save_file("/nonexistent_dir/seq_state.bin", 0, &[]);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_load_file_with_zero_max_tokens_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_session_zero_max.bin");
        context.state_save_file(&session_path, &tokens).unwrap();

        let result = context.state_load_file(&session_path, 0);

        assert!(result.is_err());
        let _ = std::fs::remove_file(&session_path);
    }

    #[test]
    #[serial]
    fn state_seq_load_file_with_zero_max_tokens_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_seq_state_zero_max.bin");
        context
            .state_seq_save_file(&session_path, 0, &tokens)
            .unwrap();

        let result = context.state_seq_load_file(&session_path, 0, 0);

        assert!(result.is_err());
        let _ = std::fs::remove_file(&session_path);
    }

    #[test]
    #[serial]
    fn state_load_file_with_insufficient_max_tokens_returns_length_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model
            .str_to_token(
                "Hello world this is a longer string for more tokens",
                AddBos::Always,
            )
            .unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_session_insuf.bin");
        context.state_save_file(&session_path, &tokens).unwrap();

        let result = context.state_load_file(&session_path, 1);

        assert!(result.is_err());
        let _ = std::fs::remove_file(&session_path);
    }

    #[test]
    #[serial]
    fn state_seq_load_file_with_insufficient_max_tokens_returns_length_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model
            .str_to_token(
                "Hello world this is a longer string for more tokens",
                AddBos::Always,
            )
            .unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let session_path = std::env::temp_dir().join("llama_test_seq_state_insuf.bin");
        context
            .state_seq_save_file(&session_path, 0, &tokens)
            .unwrap();

        let result = context.state_seq_load_file(&session_path, 0, 1);

        assert!(result.is_err());
        let _ = std::fs::remove_file(&session_path);
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn state_save_file_with_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
        let result = context.state_save_file(non_utf8_path, &[]);

        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn state_load_file_with_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
        let result = context.state_load_file(non_utf8_path, 512);

        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn state_seq_save_file_with_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
        let result = context.state_seq_save_file(non_utf8_path, 0, &[]);

        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn state_seq_load_file_with_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.bin"));
        let result = context.state_seq_load_file(non_utf8_path, 0, 512);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_save_file_with_null_byte_in_path_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
        let result = context.state_save_file(path_with_null, &[]);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_load_file_with_null_byte_in_path_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
        let result = context.state_load_file(path_with_null, 512);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_seq_save_file_with_null_byte_in_path_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();

        let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
        let result = context.state_seq_save_file(path_with_null, 0, &[]);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_seq_load_file_with_null_byte_in_path_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let path_with_null = std::path::Path::new("/tmp/foo\0bar.bin");
        let result = context.state_seq_load_file(path_with_null, 0, 512);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn state_seq_get_size_ext_returns_size_for_decoded_sequence() {
        use crate::context::llama_state_seq_flags::LlamaStateSeqFlags;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let flags = LlamaStateSeqFlags::empty();
        let size = context.state_seq_get_size_ext(0, &flags);

        assert!(size > 0);
    }

    #[test]
    #[serial]
    fn state_seq_get_data_ext_and_set_data_ext_round_trip() {
        use crate::context::llama_state_seq_flags::LlamaStateSeqFlags;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello world", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let flags = LlamaStateSeqFlags::empty();
        let size = context.state_seq_get_size_ext(0, &flags);
        let mut buffer = vec![0u8; size];
        let bytes_written = unsafe { context.state_seq_get_data_ext(&mut buffer, 0, &flags) };

        assert!(bytes_written > 0);

        let bytes_read = unsafe { context.state_seq_set_data_ext(&buffer, 0, &flags) };

        assert!(bytes_read > 0);
    }
}
