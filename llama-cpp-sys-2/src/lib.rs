//! See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
