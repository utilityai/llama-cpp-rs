//! # Raw bindings to `llama.cpp`
//!
//! See [llama-cpp-2](https://docs.rs/llama-cpp-2/) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]

// These files are generated with `cargo run --bin generate-bindings`
#[cfg(feature = "common")]
mod common;
mod ggml;
mod gguf;
mod llama;
#[cfg(feature = "mtmd")]
mod mtmd;

#[cfg(feature = "common")]
pub use self::common::*;
pub use self::ggml::*;
pub use self::gguf::*;
pub use self::llama::*;
#[cfg(feature = "mtmd")]
pub use self::mtmd::*;

/// Use a relatively decent cross-platform definition for `FILE`.
///
/// We could use `libc::FILE` here too, but that'd introduce a dependency that
/// we don't really need.
pub(crate) type FILE = std::os::raw::c_void;
