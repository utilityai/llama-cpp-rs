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

// See generate-bindings for more information on this.
#[cfg(any(all(target_os = "windows", target_env = "msvc"), target_os = "uefi"))]
pub(crate) type UnsignedEnum = std::os::raw::c_int;
#[cfg(target_arch = "hexagon")]
pub(crate) type UnsignedEnum = std::os::raw::c_uchar;
#[cfg(not(any(
    all(target_os = "windows", target_env = "msvc"),
    target_os = "uefi",
    target_arch = "hexagon",
)))]
pub(crate) type UnsignedEnum = std::os::raw::c_uint;

// See generate-bindings for more information on this.
#[cfg(target_arch = "hexagon")]
pub(crate) type SignedEnum = std::os::raw::c_schar;
#[cfg(not(target_arch = "hexagon",))]
pub(crate) type SignedEnum = std::os::raw::c_int;

extern "C" {
    pub fn ggml_abort(
        file: *const std::os::raw::c_char,
        line: std::os::raw::c_int,
        fmt: *const std::os::raw::c_char,
        ...
    ) -> !;
}

#[cfg(test)]
mod tests {
    #[test]
    fn cxx_stdlib_link_directive_is_not_duplicated() {
        let out_dir = env!("OUT_DIR");
        let output_path = std::path::Path::new(out_dir)
            .parent()
            .expect("OUT_DIR should have a parent build directory")
            .join("output");
        let output = std::fs::read_to_string(&output_path).unwrap_or_else(|err| {
            panic!(
                "failed to read build script output {}: {err}",
                output_path.display()
            );
        });
        let cxx_link_count = output
            .lines()
            .filter(|line| *line == "cargo:rustc-link-lib=c++")
            .count();
        assert!(
            cxx_link_count <= 1,
            "build.rs emitted cargo:rustc-link-lib=c++ {cxx_link_count} times; \
             the cc crate auto-link plus the explicit Apple println must not both fire"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(
            cxx_link_count, 1,
            "macOS must still link libc++ exactly once via the explicit Apple directive"
        );
    }
}
