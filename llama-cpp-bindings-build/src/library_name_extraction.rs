use std::path::Path;

use glob::glob;

use crate::debug_log;

fn extract_single_lib_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;

    if let Some(stripped) = stem.strip_prefix("lib") {
        return Some(stripped.to_string());
    }

    if path.extension() == Some(std::ffi::OsStr::new("a"))
        && let Some(parent) = path.parent()
    {
        let renamed_path = parent.join(format!("lib{stem}.a"));

        if let Err(error) = std::fs::rename(path, &renamed_path) {
            println!(
                "cargo:warning=failed to rename {} to {}: {error}",
                path.display(),
                renamed_path.display()
            );
        }
    }

    Some(stem.to_string())
}

pub fn extract_lib_names(cmake_dir: &Path, build_shared_libs: bool) -> Vec<String> {
    let lib_pattern = if cfg!(windows) {
        "*.lib"
    } else if cfg!(target_os = "macos") {
        if build_shared_libs { "*.dylib" } else { "*.a" }
    } else if build_shared_libs {
        "*.so"
    } else {
        "*.a"
    };

    let libs_dir = cmake_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let pattern_str = pattern.to_string_lossy();
    let mut lib_names: Vec<String> = Vec::new();

    let Ok(entries) = glob(&pattern_str) else {
        println!("cargo:warning=failed to glob library pattern: {pattern_str}");

        return lib_names;
    };

    for entry in entries {
        match entry {
            Ok(path) => {
                if let Some(lib_name) = extract_single_lib_name(&path) {
                    lib_names.push(lib_name);
                }
            }
            Err(error) => println!("cargo:warning=glob error: {error}"),
        }
    }

    lib_names
}
