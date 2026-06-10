use std::path::Path;

use crate::glob_paths;
use crate::target_os::TargetOs;

const WRAPPER_SOURCE_PATTERNS: &[&str] = &["wrapper_*.cpp"];

pub fn compile_cpp_wrappers(llama_src: &Path, target_os: &TargetOs) {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .include(".")
        .include("GSL/include")
        .include(llama_src)
        .include(llama_src.join("common"))
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    for pattern in WRAPPER_SOURCE_PATTERNS {
        match glob_paths::collect_paths(pattern) {
            Ok(paths) => {
                for path in paths {
                    build.file(&path);
                }
            }
            Err(error) => panic!("cpp wrapper discovery failed: {error}"),
        }
    }

    if target_os.is_msvc() {
        build.flag("/std:c++17");
        build.flag("/EHsc");
    }

    if target_os.is_android() && cfg!(feature = "static-stdcxx") {
        build.cpp_link_stdlib(None);
    }

    build.compile("llama_cpp_bindings_sys_common_wrapper");
}
