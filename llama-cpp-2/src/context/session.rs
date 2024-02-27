//! utilities for working with session files

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};

/// Failed to save a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    /// llama.cpp failed to save the session file
    #[error("Failed to save session file")]
    FailedToSave,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// Failed to load a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSessionError {
    /// llama.cpp failed to load the session file
    #[error("Failed to load session file")]
    FailedToLoad,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

impl LlamaContext<'_> {
    /// Save the current session to a file.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to save to.
    /// * `tokens` - The tokens to associate the session with. This should be a prefix of a sequence of tokens that the context has processed, so that the relevant KV caches are already filled.
    /// 
    /// # Errors
    /// 
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to save the session file.
    pub fn save_session_file(
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
            llama_cpp_sys_2::llama_save_session_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens.as_ptr() as *const i32,
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }
    /// Load a session file into the current context.
    ///
    /// You still need to pass the returned tokens to the context for inference to work. What this function buys you is that the KV caches are already filled with the relevant data.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to load from. It must be a session file from a compatible context, otherwise the function will error.
    /// * `max_tokens` - The maximum token length of the loaded session. If the session was saved with a longer length, the function will error.
    /// 
    /// # Errors
    /// 
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to load the session file. (e.g. the file does not exist, is not a session file, etc.)
    pub fn load_session_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, LoadSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or(LoadSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        unsafe {
            if llama_cpp_sys_2::llama_load_session_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens.as_mut_ptr().cast::<i32>(),
                max_tokens,
                &mut n_out,
            ) {
                assert!(n_out <= max_tokens, "n_out is greater than max_tokens");
                tokens.set_len(n_out);
                Ok(tokens)
            } else {
                Err(LoadSessionError::FailedToLoad)
            }
        }
    }
}
