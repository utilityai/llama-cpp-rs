use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;

use crate::BuildContext;
use crate::android_ndk::AndroidNdk;
use crate::cmake_common;
use crate::ggml_cmake_options;
use crate::target_os::{TargetOs, WindowsVariant};

pub fn configure_and_build(context: &BuildContext) -> PathBuf {
    let mut config = Config::new(&context.llama_src);

    configure_base_defines(&mut config);
    cmake_common::pass_cmake_env_vars(&mut config);
    cmake_common::configure_compiler_launchers(&mut config);
    ggml_cmake_options::configure_cpu_features(&mut config, &context.target_triple);
    cmake_common::configure_shared_libs(&mut config, context.build_shared_libs);
    configure_platform_specific(
        &mut config,
        &context.target_os,
        &context.target_triple,
        &context.profile,
        context.android_ndk.as_ref(),
    );
    ggml_cmake_options::configure_gpu_backends(&mut config, &context.target_os);
    ggml_cmake_options::configure_openmp(&mut config, &context.target_os);
    configure_system_ggml(&mut config);

    if let Some(ggml) = context.ggml_system.as_ref() {
        config.define(
            "ggml_DIR",
            ggml.cmake_dir
                .to_str()
                .expect("ggml cmake directory must be valid UTF-8"),
        );
    }

    let backends_dir = configure_dynamic_backends(&mut config, &context.cmake_dir);

    config.static_crt(context.static_crt);
    config
        .out_dir(&context.cmake_dir)
        .profile(&context.profile)
        .very_verbose(env::var("CMAKE_VERBOSE").is_ok())
        .always_configure(false);

    let install_dir = config.build();

    if let Some(dir) = backends_dir {
        println!("cargo:backends_dir={}", dir.display());
    }

    install_dir
}

fn configure_dynamic_backends(config: &mut Config, cmake_dir: &Path) -> Option<PathBuf> {
    if !cfg!(feature = "dynamic-backends") {
        return None;
    }

    let backends_dir = cmake_dir.join("backends");

    std::fs::create_dir_all(&backends_dir).expect("failed to create backends directory");

    config.define("GGML_BACKEND_DL", "ON");
    config.define("GGML_CPU_ALL_VARIANTS", "ON");
    config.define(
        "GGML_BACKEND_DIR",
        backends_dir
            .to_str()
            .expect("backends directory must be valid UTF-8"),
    );

    Some(backends_dir)
}

fn configure_base_defines(config: &mut Config) {
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    config.define("LLAMA_BUILD_APP", "OFF");
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_CURL", "OFF");
    config.cflag("-w");
    config.cxxflag("-w");
}

fn configure_platform_specific(
    config: &mut Config,
    target_os: &TargetOs,
    target_triple: &str,
    profile: &str,
    android_ndk: Option<&AndroidNdk>,
) {
    match target_os {
        TargetOs::Apple(_) => {
            config.define("GGML_BLAS", "OFF");
            cmake_common::override_archive_commands_for_apple_ar(config);
        }
        TargetOs::Windows(WindowsVariant::Msvc) => {
            config.cflag("/w");
            config.cxxflag("/w");
            config.cxxflag("/EHsc");
            configure_msvc_release_workaround(config, profile);
        }
        TargetOs::Android => {
            if let Some(ndk) = android_ndk {
                configure_android_cmake(config, ndk, target_triple);
            }
        }
        _ => {}
    }
}

fn configure_msvc_release_workaround(config: &mut Config, profile: &str) {
    let is_release_profile = matches!(profile, "Release" | "RelWithDebInfo" | "MinSizeRel");

    if !is_release_profile {
        return;
    }

    for flag in &["/O2", "/DNDEBUG", "/Ob2"] {
        config.cflag(flag);
        config.cxxflag(flag);
    }
}

fn configure_android_cmake(config: &mut Config, ndk: &AndroidNdk, _target_triple: &str) {
    #[cfg(all(feature = "shared-stdcxx", feature = "static-stdcxx"))]
    compile_error!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");

    println!("cargo:rerun-if-env-changed=ANDROID_NDK");
    println!("cargo:rerun-if-env-changed=NDK_ROOT");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=ANDROID_PLATFORM");
    println!("cargo:rerun-if-env-changed=ANDROID_API_LEVEL");

    config.define("CMAKE_TOOLCHAIN_FILE", ndk.cmake_toolchain_file());
    config.define("ANDROID_PLATFORM", ndk.android_platform());
    config.define("ANDROID_ABI", ndk.abi);

    if cfg!(feature = "static-stdcxx") {
        config.define("ANDROID_STL", "c++_static");
    } else if cfg!(feature = "shared-stdcxx") {
        config.define("ANDROID_STL", "c++_shared");
    }

    configure_android_arch_flags(config, ndk.abi);

    config.define("GGML_LLAMAFILE", "OFF");

    println!("cargo:rustc-link-lib=log");
    println!("cargo:rustc-link-lib=android");
}

fn configure_android_arch_flags(config: &mut Config, abi: &str) {
    match abi {
        "arm64-v8a" => {
            config.cflag("-march=armv8-a");
            config.cxxflag("-march=armv8-a");
        }
        "armeabi-v7a" => {
            config.cflag("-march=armv7-a");
            config.cxxflag("-march=armv7-a");
            config.cflag("-mfpu=neon");
            config.cxxflag("-mfpu=neon");
            config.cflag("-mthumb");
            config.cxxflag("-mthumb");
        }
        "x86_64" => {
            config.cflag("-march=x86-64");
            config.cxxflag("-march=x86-64");
        }
        "x86" => {
            config.cflag("-march=i686");
            config.cxxflag("-march=i686");
        }
        _ => {}
    }
}

fn configure_system_ggml(config: &mut Config) {
    if cfg!(feature = "system-ggml") {
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
    }
}
