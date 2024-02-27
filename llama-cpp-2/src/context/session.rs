//! utilities for working with session files

use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};
use crate::context::LlamaContext;
use crate::token::LlamaToken;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    #[error("Failed to save session file")]
    FailedToSave,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSessionError {
    #[error("Failed to load session file")]
    FailedToLoad,

    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

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
    pub fn save_session_file(&self, path_session: impl AsRef<Path>, tokens: &[LlamaToken]) -> Result<(), SaveSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or(SaveSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        if unsafe {
            llama_cpp_sys_2::llama_save_session_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens.as_ptr() as *const i32,
                tokens.len())
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }
    /// Load a session file into the current context.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to load from. It must be a session file from a compatible context, otherwise the function will error.
    /// * `max_tokens` - The maximum token length of the loaded session. If the session was saved with a longer length, the function will error.
    pub fn load_session_file(&mut self, path_session: impl AsRef<Path>, max_tokens: usize) -> Result<Vec<LlamaToken>, LoadSessionError> {
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
                tokens.as_mut_ptr() as *mut i32,
                max_tokens,
                &mut n_out) {
                tokens.set_len(n_out);
                Ok(tokens)
            } else {
                Err(LoadSessionError::FailedToLoad)
            }
        }
    }
}