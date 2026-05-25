#![expect(
    non_camel_case_types,
    reason = "bindgen emits C struct and enum names verbatim and they don't follow Rust naming"
)]
#![expect(
    unpredictable_function_pointer_comparisons,
    reason = "bindgen-generated FFI function pointers are opaque and the lint cannot reason about them"
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
