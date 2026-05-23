//! See [llama-cpp-bindings](https://crates.io/crates/llama-cpp-bindings) for a documented and safe API.

#![expect(
    non_camel_case_types,
    reason = "bindgen emits C struct and enum names verbatim and they don't follow Rust naming"
)]
#![expect(
    non_snake_case,
    reason = "bindgen emits C function names verbatim and they don't always follow Rust naming"
)]
#![expect(
    unpredictable_function_pointer_comparisons,
    reason = "bindgen-generated FFI function pointers are opaque and the lint cannot reason about them"
)]
#![expect(
    unnecessary_transmutes,
    reason = "bindgen generates transmutes to bridge between C and Rust integer/enum representations"
)]
#![expect(
    clippy::missing_safety_doc,
    reason = "bindgen emits raw FFI declarations; safety contracts live on the wrapper API in llama-cpp-bindings"
)]
#![expect(
    clippy::ptr_offset_with_cast,
    reason = "bindgen emits standard FFI pointer-arithmetic patterns that this lint flags"
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
