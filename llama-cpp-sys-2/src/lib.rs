//! See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Compile-time path to the built GGML backend modules directory.
/// Populated by build.rs from `GGML_BACKENDS_DIR`.
/// None on static builds or when the feature is disabled.
pub const BACKENDS_DIR: Option<&str> = option_env!("GGML_BACKENDS_DIR");
