//! Safe wrappers around `llama_token_data` and `llama_token_data_array`.

use std::fmt::Debug;
use std::fmt::Display;

pub mod data;
pub mod data_array;

/// A safe wrapper for `llama_token`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaToken(pub llama_cpp_sys_2::llama_token);

impl Display for LlamaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LlamaToken {
    /// Create a new `LlamaToken` from a i32.
    ///
    /// ```
    /// # use llama_cpp_2::token::LlamaToken;
    /// let token = LlamaToken::new(0);
    /// assert_eq!(token, LlamaToken(0));
    /// ```
    #[must_use]
    pub fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}

/// convert a vector of `llama_token` to a vector of `LlamaToken` without memory allocation, 
/// and consume the original vector.
/// SAFETY: cast is valid as LlamaToken is repr(transparent)
pub fn from_vec_token_sys(mut vec_sys: Vec<llama_cpp_sys_2::llama_token>) -> Vec<LlamaToken> {
    let ptr = vec_sys.as_mut_ptr() as *mut LlamaToken;
    unsafe {
        Vec::from_raw_parts(ptr, vec_sys.len(), vec_sys.capacity())
    }
}

/// convert a vector of `LlamaToken` to a vector of `llama_token` without memory allocation, 
/// and consume the original vector.
/// SAFETY: cast is valid as LlamaToken is repr(transparent)
pub fn to_vec_token_sys(mut vec_llama: Vec<LlamaToken>) -> Vec<llama_cpp_sys_2::llama_token> {
    let ptr = vec_llama.as_mut_ptr() as *mut llama_cpp_sys_2::llama_token;
    unsafe {
        Vec::from_raw_parts(ptr, vec_llama.len(), vec_llama.capacity())
    }
}

