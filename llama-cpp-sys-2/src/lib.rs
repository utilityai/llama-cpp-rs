//! See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// [`ash`] is only included to link to the Vulkan SDK.
#[allow(unused)]
#[cfg(feature = "vulkan")]
use ash;

// [`cudarc`] is only included to link to CUDA.
#[allow(unused)]
#[cfg(feature = "cuda")]
use cudarc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
