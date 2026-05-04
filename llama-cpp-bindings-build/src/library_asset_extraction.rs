use std::path::{Path, PathBuf};

use glob::glob;

use crate::debug_log;

pub fn extract_lib_assets(cmake_dir: &Path) -> Vec<PathBuf> {
    let shared_lib_pattern = if cfg!(windows) {
        "*.dll"
    } else if cfg!(target_os = "macos") {
        "*.dylib"
    } else {
        "*.so"
    };

    let shared_libs_dir = if cfg!(windows) { "bin" } else { "lib" };
    let libs_dir = cmake_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());

    let pattern_str = pattern.to_string_lossy();
    let mut files = Vec::new();

    let Ok(entries) = glob(&pattern_str) else {
        println!("cargo:warning=failed to glob shared lib pattern: {pattern_str}");

        return files;
    };

    for entry in entries {
        match entry {
            Ok(path) => files.push(path),
            Err(error) => eprintln!("cargo:warning=glob error: {error}"),
        }
    }

    files
}
