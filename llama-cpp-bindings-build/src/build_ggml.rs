use std::env;
use std::path::Path;
use std::path::PathBuf;

use cmake::Config;

use crate::cmake_common;
use crate::ggml_cmake_options;
use crate::target_os::TargetOs;

const GGML_SUBMODULE_RELATIVE_PATH: &str = "../llama-cpp-bindings-sys/llama.cpp/ggml";

pub fn build_ggml() {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR env var is required");
    let target_triple = env::var("TARGET").expect("TARGET env var is required in build scripts");
    let target_os = TargetOs::from_target_triple(&target_triple)
        .unwrap_or_else(|error| panic!("Failed to parse target OS: {error}"));
    let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or_else(|_| "Release".to_string());

    let wrapper_cmake_dir = PathBuf::from(&manifest_dir);
    let ggml_src = wrapper_cmake_dir.join(GGML_SUBMODULE_RELATIVE_PATH);

    assert!(
        ggml_src.join("CMakeLists.txt").is_file(),
        "ggml source not found at {}; ensure the llama.cpp submodule is checked out",
        ggml_src.display()
    );

    register_rebuild_triggers(&ggml_src);

    let install_dir = configure_and_install_ggml(
        &wrapper_cmake_dir,
        &ggml_src,
        &target_triple,
        &target_os,
        &profile,
    );

    emit_metadata(&install_dir);
}

fn configure_and_install_ggml(
    wrapper_cmake_dir: &Path,
    ggml_src: &Path,
    target_triple: &str,
    target_os: &TargetOs,
    profile: &str,
) -> PathBuf {
    let mut config = Config::new(wrapper_cmake_dir);

    config.define(
        "GGML_SOURCE_DIR",
        ggml_src
            .to_str()
            .expect("ggml source path must be valid UTF-8"),
    );
    config.define("GGML_BUILD_TESTS", "OFF");
    config.define("GGML_BUILD_EXAMPLES", "OFF");
    config.cflag("-w");
    config.cxxflag("-w");

    cmake_common::pass_cmake_env_vars(&mut config);
    cmake_common::configure_compiler_launchers(&mut config);
    cmake_common::configure_shared_libs(&mut config, false);
    ggml_cmake_options::configure_cpu_features(&mut config, target_triple);
    ggml_cmake_options::configure_gpu_backends(&mut config, target_os);
    ggml_cmake_options::configure_openmp(&mut config, target_os);

    if let TargetOs::Apple(_) = target_os {
        config.define("GGML_BLAS", "OFF");
        cmake_common::override_archive_commands_for_apple_ar(&mut config);
    }

    config
        .profile(profile)
        .very_verbose(env::var("CMAKE_VERBOSE").is_ok())
        .always_configure(false);

    config.build()
}

fn emit_metadata(install_dir: &Path) {
    let lib_dir = resolve_lib_dir(install_dir);

    println!("cargo:root={}", install_dir.display());
    println!("cargo:include={}", install_dir.join("include").display());
    println!("cargo:lib={}", lib_dir.display());
    println!(
        "cargo:cmake={}",
        lib_dir.join("cmake").join("ggml").display()
    );
}

fn resolve_lib_dir(install_dir: &Path) -> PathBuf {
    let lib64 = install_dir.join("lib64");

    if lib64.is_dir() {
        lib64
    } else {
        install_dir.join("lib")
    }
}

fn register_rebuild_triggers(ggml_src: &Path) {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_DISABLE_CCACHE");
    println!("cargo:rerun-if-env-changed=CMAKE_VERBOSE");
    println!(
        "cargo:rerun-if-changed={}",
        ggml_src.join("CMakeLists.txt").display()
    );
    println!("cargo:rerun-if-changed={}", ggml_src.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        ggml_src.join("include").display()
    );
}
