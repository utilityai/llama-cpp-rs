use std::path::Path;

use crate::glob_paths;
use crate::target_os::TargetOs;

const MTMD_SKIP_FILES: &[&str] = &["mtmd-cli.cpp", "deprecation-warning.cpp"];

pub fn compile_mtmd(llama_src: &Path, target_os: &TargetOs) {
    let mtmd_src = llama_src.join("tools/mtmd");
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .include(&mtmd_src)
        .include(llama_src)
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("common"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    if target_os.is_msvc() {
        build.flag("/std:c++17");
        build.flag("/EHsc");
    }

    if target_os.is_android() && cfg!(feature = "static-stdcxx") {
        build.cpp_link_stdlib(None);
    }

    let pattern = mtmd_src.join("**/*.cpp");
    let pattern_str = pattern.to_string_lossy();

    let paths = match glob_paths::collect_paths(&pattern_str) {
        Ok(paths) => paths,
        Err(error) => panic!("mtmd source discovery failed: {error}"),
    };

    for path in paths {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();

        if MTMD_SKIP_FILES.contains(&filename) {
            continue;
        }

        build.file(&path);
    }

    build.compile("mtmd");
}
