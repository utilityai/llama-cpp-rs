mod android_ndk;
mod bindgen_config;
pub mod build_gbnf;
pub mod build_ggml;
mod cmake_common;
mod cmake_config;
mod cpp_wrapper;
mod cpp_wrapper_mtmd;
mod ggml_cmake_options;
mod ggml_system_paths;
mod glob_paths;
mod library_asset_extraction;
mod library_linking;
mod library_name_extraction;
mod rebuild_tracking;
mod shared_libs;
mod stable_cmake_build_dir;
mod target_os;

use std::env;
use std::path::{Path, PathBuf};

use android_ndk::AndroidNdk;
use ggml_system_paths::GgmlSystemPaths;
use stable_cmake_build_dir::stable_cmake_build_dir;
use target_os::TargetOs;

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

#[derive(Debug)]
pub struct BuildContext {
    pub out_dir: PathBuf,
    pub target_dir: PathBuf,
    pub cmake_dir: PathBuf,
    pub llama_src: PathBuf,
    pub target_os: TargetOs,
    pub target_triple: String,
    pub build_shared_libs: bool,
    pub profile: String,
    pub static_crt: bool,
    pub android_ndk: Option<AndroidNdk>,
    pub ggml_system: Option<GgmlSystemPaths>,
}

impl BuildContext {
    fn detect() -> Self {
        let target_triple =
            env::var("TARGET").expect("TARGET env var is required in build scripts");
        let target_os = TargetOs::from_target_triple(&target_triple)
            .unwrap_or_else(|error| panic!("Failed to parse target OS: {error}"));
        let out_dir = PathBuf::from(
            env::var("OUT_DIR").expect("OUT_DIR env var is required in build scripts"),
        );
        let target_dir = cargo_target_dir(&out_dir);
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR env var is required in build scripts");
        let llama_src = Path::new(&manifest_dir).join("llama.cpp");

        let build_shared_libs = env::var("LLAMA_BUILD_SHARED_LIBS")
            .map_or_else(|_| cfg!(feature = "dynamic-link"), |value| value == "1");

        let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or_else(|_| "Release".to_string());

        let static_crt = env::var("LLAMA_STATIC_CRT")
            .map(|value| value == "1")
            .unwrap_or(false);

        let android_ndk = if target_os.is_android() {
            Some(
                AndroidNdk::detect(&target_triple)
                    .unwrap_or_else(|error| panic!("Android NDK detection failed: {error}")),
            )
        } else {
            None
        };

        let ggml_system = if cfg!(feature = "system-ggml") {
            Some(GgmlSystemPaths::from_env())
        } else {
            None
        };

        let cmake_dir = stable_cmake_build_dir(
            &target_dir,
            &target_triple,
            &profile,
            static_crt,
            build_shared_libs,
        );

        debug_log!("TARGET: {}", target_triple);
        debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
        debug_log!("TARGET_DIR: {}", target_dir.display());
        debug_log!("OUT_DIR: {}", out_dir.display());
        debug_log!("CMAKE_DIR: {}", cmake_dir.display());
        debug_log!("BUILD_SHARED: {}", build_shared_libs);

        Self {
            out_dir,
            target_dir,
            cmake_dir,
            llama_src,
            target_os,
            target_triple,
            build_shared_libs,
            profile,
            static_crt,
            android_ndk,
            ggml_system,
        }
    }
}

fn cargo_target_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR is not deep enough to determine target directory")
        .to_path_buf()
}

fn set_cmake_parallelism() {
    if let Ok(parallelism) = std::thread::available_parallelism() {
        // SAFETY: build scripts are single-threaded, so modifying env is safe.
        unsafe {
            env::set_var("CMAKE_BUILD_PARALLEL_LEVEL", parallelism.get().to_string());
        }
    }
}

pub fn build() {
    let context = BuildContext::detect();

    rebuild_tracking::register_rebuild_triggers(&context.llama_src);

    set_cmake_parallelism();

    bindgen_config::generate_bindings(
        &context.llama_src,
        &context.out_dir,
        &context.target_os,
        &context.target_triple,
        context.android_ndk.as_ref(),
    );

    cpp_wrapper::compile_cpp_wrappers(&context.llama_src, &context.target_os);

    let build_dir = cmake_config::configure_and_build(&context);

    cpp_wrapper_mtmd::compile_mtmd(&context.llama_src, &context.target_os);

    library_linking::link_libraries(
        &context.cmake_dir,
        &build_dir,
        &context.target_os,
        &context.target_triple,
        context.build_shared_libs,
        &context.profile,
        context.ggml_system.as_ref(),
    );

    if context.build_shared_libs {
        shared_libs::copy_shared_libraries(&context.cmake_dir, &context.target_dir);
    }
}
