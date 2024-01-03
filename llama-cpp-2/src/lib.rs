//! Bindings to the llama.cpp library.
//!
//! As llama.cpp is a very fast moving target, this crate does not attempt to create a stable API
//! with all the rust idioms. Instead it provided safe wrappers around nearly direct bindings to
//! llama.cpp. This makes it easier to keep up with the changes in llama.cpp, but does mean that
//! the API is not as nice as it could be.
//!
//! # Examples
//!
//! ## Inference
//!
//! ```no_run
//! use llama_cpp_2::model::LlamaModel;
//! use llama_cpp_2::llama_backend::LlamaBackend;
//! use llama_cpp_2::context::params::LlamaContextParams;
//! use llama_cpp_2::llama_batch::LlamaBatch;
//! use llama_cpp_2::model::params::LlamaModelParams;
//! use llama_cpp_2::token::data_array::LlamaTokenDataArray;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // initialize GGML
//! let backend = LlamaBackend::init()?;
//!
//! // load the model (this may be slow)
//! let model = LlamaModel::load_from_file(&backend, "path/to/model", &LlamaModelParams::default())?;
//! let prompt = "How do I kill a process on linux?";
//! let tokens = model.str_to_token(prompt, true)?;
//!
//! // create a context and batch
//! let mut context = model.new_context(&backend, &LlamaContextParams::default())?;
//! let mut batch = LlamaBatch::new(512, 1);
//! let mut pos: i32 = 0;
//!
//! // add the prompt to the batch
//! let last_index = i32::try_from(tokens.len()).unwrap() - 1;
//! for token in tokens {
//!     batch.add(token, pos, &[0], pos == last_index);
//!     pos += 1;
//! }
//!
//! let mut response = vec![];
//!
//! // evaluate first 10 tokens
//! for i in 0..10 {
//!     context.decode(&mut batch)?;
//!     let token = context.sample_token_greedy(LlamaTokenDataArray::from_iter(context.candidates_ith(batch.n_tokens()), false));
//!     response.push(token);
//!     batch.clear();
//! }
//!
//! let response_str = model.tokens_to_str(&response)?;
//! println!("{}", response_str);
//!
//! # Ok(())
//! # }
use std::ffi::NulError;
use std::fmt::Debug;
use std::num::NonZeroI32;

use std::os::raw::c_int;
use std::path::PathBuf;
use std::string::FromUtf8Error;

pub mod context;
pub mod grammar;
pub mod llama_backend;
pub mod llama_batch;
pub mod model;
pub mod timing;
pub mod token;
pub mod token_type;

/// A failable result from a llama.cpp function.
pub type Result<T> = std::result::Result<T, LLamaCppError>;

/// All errors that can occur in the llama-cpp crate.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LLamaCppError {
    /// The backend was already initialized. This can generally be ignored as initializing the backend
    /// is idempotent.
    #[error("BackendAlreadyInitialized")]
    BackendAlreadyInitialized,
    /// There was an error while decoding a batch.
    #[error("{0}")]
    DecodeError(#[from] DecodeError),
    /// There was an error loading a model.
    #[error("{0}")]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    /// There was an error creating a new model context.
    #[error("{0}")]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
}

/// Failed to Load context
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    /// llama.cpp returned null
    #[error("null reference from llama.cpp")]
    NullReturn,
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// No kv cache slot was available.
    #[error("Decode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Decode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Decode Error {0}: unknown")]
    Unknown(c_int),
}

/// Decode a error from llama.cpp into a [`DecodeError`].
impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => DecodeError::NoKvCacheSlot,
            -1 => DecodeError::NTokensZero,
            i => DecodeError::Unknown(i),
        }
    }
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaModelLoadError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// get the time (in microseconds) according to llama.cpp
/// ```
/// # use llama_cpp_2::llama_time_us;
/// let time = llama_time_us();
/// assert!(time > 0);
/// ```
#[must_use]
pub fn llama_time_us() -> i64 {
    unsafe { llama_cpp_sys_2::llama_time_us() }
}

/// get the max number of devices according to llama.cpp (this is generally cuda devices)
/// ```
/// # use llama_cpp_2::max_devices;
/// let max_devices = max_devices();
/// assert!(max_devices >= 0);
/// ```
#[must_use]
pub fn max_devices() -> c_int {
    unsafe { llama_cpp_sys_2::llama_max_devices() }
}

/// is memory mapping supported according to llama.cpp
/// ```
/// # use llama_cpp_2::mmap_supported;
/// let mmap_supported = mmap_supported();
/// if mmap_supported {
///   println!("mmap_supported!");
/// }
/// ```
#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_sys_2::llama_mmap_supported() }
}

/// is memory locking supported according to llama.cpp
/// ```
/// # use llama_cpp_2::mlock_supported;
/// let mlock_supported = mlock_supported();
/// if mlock_supported {
///    println!("mlock_supported!");
/// }
/// ```
#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_sys_2::llama_mlock_supported() }
}

/// An error that can occur when converting a token to a string.
#[derive(Debug, thiserror::Error, Clone)]
#[non_exhaustive]
pub enum TokenToStringError {
    /// the token type was unknown
    #[error("Unknown Token Type")]
    UnknownTokenType,
    /// There was insufficient buffer space to convert the token to a string.
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    /// The token was not valid utf8.
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
}

/// Failed to convert a string to a token sequence.
#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    /// Failed to convert a provided integer to a c_int.
    CIntConversionError(#[from] std::num::TryFromIntError),
}
