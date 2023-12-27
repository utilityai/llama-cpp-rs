//! Safe wrappers around `llama_token_data` and `llama_token_data_array`.

use std::fmt::Debug;

pub mod data;
pub mod data_array;

/// A safe wrapper for `llama_token`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaToken(pub llama_cpp_sys::llama_token);

impl LlamaToken {
    /// Create a new `LlamaToken` from a i32.
    ///
    /// ```
    /// # use llama_cpp::token::LlamaToken;
    /// let token = LlamaToken::new(0);
    /// assert_eq!(token, LlamaToken(0));
    /// ```
    #[must_use]
    pub fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}
