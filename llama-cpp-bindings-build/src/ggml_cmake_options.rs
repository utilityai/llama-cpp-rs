use std::env;
use std::path::Path;

use cmake::Config;

use crate::debug_log;
use crate::target_os::TargetOs;

pub fn configure_cpu_features(config: &mut Config, target_triple: &str) {
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

pub fn configure_gpu_backends(config: &mut Config, target_os: &TargetOs) {
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

pub fn configure_openmp(config: &mut Config, target_os: &TargetOs) {
    let openmp_enabled = cfg!(feature = "openmp") && !target_os.is_android();

    config.define("GGML_OPENMP", if openmp_enabled { "ON" } else { "OFF" });
}
