use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;

use crate::BuildContext;
use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::target_os::{TargetOs, WindowsVariant};

pub fn configure_and_build(context: &BuildContext) -> PathBuf {
    let mut config = Config::new(&context.llama_src);

    configure_base_defines(&mut config);
    pass_cmake_env_vars(&mut config);
    configure_compiler_launchers(&mut config);
    configure_cpu_features(&mut config, &context.target_triple);
    configure_shared_libs(&mut config, context.build_shared_libs);
    configure_platform_specific(
        &mut config,
        &context.target_os,
        &context.target_triple,
        &context.profile,
        context.android_ndk.as_ref(),
    );
    configure_gpu_backends(&mut config, &context.target_os);
    configure_openmp(&mut config, &context.target_os);
    configure_system_ggml(&mut config);
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
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_CURL", "OFF");
    config.cflag("-w");
    config.cxxflag("-w");
}

fn configure_compiler_launchers(config: &mut Config) {
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

fn which(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;

    for entry in env::split_paths(&path) {
        let candidate = entry.join(program);

        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn pass_cmake_env_vars(config: &mut Config) {
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }
}

fn configure_cpu_features(config: &mut Config, target_triple: &str) {
    let target_cpu = env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .and_then(|rustflags| {
            rustflags
                .split('\x1f')
                .find(|flag| flag.contains("target-cpu="))
                .and_then(|flag| flag.split("target-cpu=").nth(1))
                .map(std::string::ToString::to_string)
        });

    if target_cpu.as_deref() == Some("native") {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");

        return;
    }

    config.define("GGML_NATIVE", "OFF");

    if let Some(ref cpu) = target_cpu {
        debug_log!("Setting baseline architecture: -march={}", cpu);
        config.cflag(format!("-march={cpu}"));
        config.cxxflag(format!("-march={cpu}"));
    }

    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    debug_log!("Compiling with target features: {}", features);

    for feature in features.split(',') {
        if let Some(ggml_flag) = map_cpu_feature_to_ggml(feature) {
            config.define(ggml_flag, "ON");
        }
    }

    if target_triple.contains("aarch64")
        && target_triple.contains("linux")
        && target_cpu.as_deref() != Some("native")
    {
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }
}

fn map_cpu_feature_to_ggml(feature: &str) -> Option<&'static str> {
    match feature {
        "avx" => Some("GGML_AVX"),
        "avx2" => Some("GGML_AVX2"),
        "avx512bf16" => Some("GGML_AVX512_BF16"),
        "avx512vbmi" => Some("GGML_AVX512_VBMI"),
        "avx512vnni" => Some("GGML_AVX512_VNNI"),
        "avxvnni" => Some("GGML_AVX_VNNI"),
        "bmi2" => Some("GGML_BMI2"),
        "f16c" => Some("GGML_F16C"),
        "fma" => Some("GGML_FMA"),
        "sse4.2" => Some("GGML_SSE42"),
        _ => {
            debug_log!(
                "Unrecognized cpu feature: '{}' - skipping GGML config for it.",
                feature
            );

            None
        }
    }
}

fn configure_shared_libs(config: &mut Config, build_shared_libs: bool) {
    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );
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
            override_archive_commands_for_apple_ar(config);
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
    #[expect(
        clippy::assertions_on_constants,
        reason = "the assertion enforces a feature flag invariant at build time"
    )]
    {
        assert!(
            !(cfg!(feature = "shared-stdcxx") && cfg!(feature = "static-stdcxx")),
            "Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive"
        );
    }

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

fn override_archive_commands_for_apple_ar(config: &mut Config) {
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

fn configure_gpu_backends(config: &mut Config, target_os: &TargetOs) {
    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        configure_vulkan_linking(config, target_os);
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");

        if cfg!(feature = "cuda-no-vmm") {
            config.define("GGML_CUDA_NO_VMM", "ON");
        }
    }

    if cfg!(feature = "rocm") {
        config.define("GGML_HIP", "ON");
    }
}

fn configure_vulkan_linking(config: &mut Config, target_os: &TargetOs) {
    match target_os {
        TargetOs::Windows(_) => {
            let vulkan_path = env::var("VULKAN_SDK")
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set");
            let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");

            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");

            // SAFETY: build scripts are single-threaded, so modifying env is safe.
            unsafe { env::set_var("TrackFileAccess", "false") };

            config.cflag("/FS");
            config.cxxflag("/FS");
        }
        TargetOs::Linux => {
            if let Ok(vulkan_path) = env::var("VULKAN_SDK") {
                let vulkan_lib_path = Path::new(&vulkan_path).join("lib");

                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            }

            println!("cargo:rustc-link-lib=vulkan");
        }
        _ => (),
    }
}

fn configure_openmp(config: &mut Config, target_os: &TargetOs) {
    let openmp_enabled = cfg!(feature = "openmp") && !target_os.is_android();

    config.define("GGML_OPENMP", if openmp_enabled { "ON" } else { "OFF" });
}

fn configure_system_ggml(config: &mut Config) {
    if cfg!(feature = "system-ggml") {
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
    }
}
