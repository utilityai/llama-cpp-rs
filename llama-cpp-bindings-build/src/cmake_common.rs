use std::env;
use std::path::PathBuf;

use cmake::Config;

use crate::debug_log;

pub fn configure_compiler_launchers(config: &mut Config) {
    println!("cargo:rerun-if-env-changed=LLAMA_DISABLE_CCACHE");

    if env::var("LLAMA_DISABLE_CCACHE").is_ok() {
        return;
    }

    let Some(ccache) = which("ccache") else {
        return;
    };

    let ccache_str = ccache.display().to_string();
    debug_log!("Using ccache for compilation: {ccache_str}");

    config.define("CMAKE_C_COMPILER_LAUNCHER", &ccache_str);
    config.define("CMAKE_CXX_COMPILER_LAUNCHER", &ccache_str);
    config.define("CMAKE_CUDA_COMPILER_LAUNCHER", &ccache_str);
}

pub fn which(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;

    for entry in env::split_paths(&path) {
        let candidate = entry.join(program);

        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

pub fn pass_cmake_env_vars(config: &mut Config) {
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }
}

pub fn configure_shared_libs(config: &mut Config, build_shared_libs: bool) {
    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );
}

pub fn override_archive_commands_for_apple_ar(config: &mut Config) {
    for language in ["C", "CXX", "OBJC", "OBJCXX"] {
        config.define(
            format!("CMAKE_{language}_ARCHIVE_CREATE"),
            "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>",
        );
        config.define(
            format!("CMAKE_{language}_ARCHIVE_APPEND"),
            "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>",
        );
        config.define(
            format!("CMAKE_{language}_ARCHIVE_FINISH"),
            "<CMAKE_RANLIB> <TARGET>",
        );
    }
}
