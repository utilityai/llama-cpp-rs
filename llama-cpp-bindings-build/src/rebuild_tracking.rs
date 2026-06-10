use std::path::Path;

use walkdir::DirEntry;

use crate::glob_paths;

const WRAPPER_TRACKING_PATTERNS: &[&str] = &["wrapper*.h", "wrapper_*.cpp"];

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with('.'))
}

fn is_cmake_file(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with("CMake"))
}

pub fn register_rebuild_triggers(llama_src: &Path) {
    println!("cargo:rerun-if-changed=build.rs");

    for pattern in WRAPPER_TRACKING_PATTERNS {
        match glob_paths::collect_paths(pattern) {
            Ok(paths) => {
                for path in paths {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
            Err(error) => panic!("wrapper rebuild tracking failed: {error}"),
        }
    }

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC_CRT");
    println!("cargo:rerun-if-env-changed=LLAMA_CMAKE_BUILD_DIR_OVERRIDE");

    let source_directories = [
        llama_src.join("src"),
        llama_src.join("ggml/src"),
        llama_src.join("common"),
    ];

    for entry in walkdir::WalkDir::new(llama_src)
        .into_iter()
        .filter_entry(|entry| !is_hidden(entry))
    {
        let Ok(entry) = entry else {
            continue;
        };

        let is_source_child = source_directories
            .iter()
            .any(|source_dir| entry.path().starts_with(source_dir));

        if is_cmake_file(&entry) || is_source_child {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
}
