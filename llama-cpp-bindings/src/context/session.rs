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
