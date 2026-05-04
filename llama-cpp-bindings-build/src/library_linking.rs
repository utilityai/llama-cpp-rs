use std::env;
use std::path::Path;

use crate::debug_log;
use crate::library_name_extraction::extract_lib_names;
use crate::target_os::{AppleVariant, TargetOs, WindowsVariant};

pub fn link_libraries(
    out_dir: &Path,
    build_dir: &Path,
    target_os: &TargetOs,
    target_triple: &str,
    build_shared_libs: bool,
    profile: &str,
) {
    emit_search_paths(out_dir, build_dir);
    link_system_ggml_paths(build_dir);
    link_cmake_built_libraries(out_dir, build_shared_libs, profile);
    link_cuda_libraries(build_shared_libs);
    link_rocm_libraries(build_shared_libs);
    link_openmp(target_triple);
    link_platform_system_libraries(target_os);
}

fn emit_search_paths(out_dir: &Path, build_dir: &Path) {
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
}

fn link_system_ggml_paths(build_dir: &Path) {
    if !cfg!(feature = "system-ggml") {
        return;
    }

    let cmake_cache = build_dir.join("build").join("CMakeCache.txt");
    let Ok(cache_contents) = std::fs::read_to_string(&cmake_cache) else {
        return;
    };

    let mut ggml_lib_dirs = std::collections::HashSet::new();

    for line in cache_contents.lines() {
        let is_ggml_library_entry = line.starts_with("GGML_LIBRARY:")
            || line.starts_with("GGML_BASE_LIBRARY:")
            || line.starts_with("GGML_CPU_LIBRARY:");

        if is_ggml_library_entry
            && let Some(lib_path) = line.split('=').nth(1)
            && let Some(parent) = Path::new(lib_path).parent()
        {
            ggml_lib_dirs.insert(parent.to_path_buf());
        }
    }

    for lib_dir in ggml_lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        debug_log!("Added system GGML library path: {}", lib_dir.display());
    }
}

fn link_cmake_built_libraries(out_dir: &Path, build_shared_libs: bool, profile: &str) {
    let link_kind = if build_shared_libs {
        "dylib"
    } else if cfg!(feature = "system-ggml-static") {
        "static"
    } else if cfg!(feature = "system-ggml") {
        "dylib"
    } else {
        "static"
    };

    let lib_names = extract_lib_names(out_dir, build_shared_libs);
    assert!(!lib_names.is_empty(), "no libraries found in build output");

    link_llama_common_internal_libraries(out_dir, profile);
    link_system_ggml_libraries(link_kind);

    for lib_name in lib_names {
        let link = format!("cargo:rustc-link-lib={link_kind}={lib_name}");
        debug_log!("LINK {link}");
        println!("{link}");
    }
}

fn link_llama_common_internal_libraries(out_dir: &Path, profile: &str) {
    let common_lib_dir = out_dir.join("build").join("common");

    if common_lib_dir.is_dir() {
        emit_search_path_with_profile(&common_lib_dir, profile);
        println!("cargo:rustc-link-lib=static=llama-common-base");
    }

    let httplib_dir = out_dir.join("build").join("vendor").join("cpp-httplib");

    if httplib_dir.is_dir() {
        emit_search_path_with_profile(&httplib_dir, profile);
        println!("cargo:rustc-link-lib=static=cpp-httplib");
    }
}

fn emit_search_path_with_profile(lib_dir: &Path, profile: &str) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let profile_dir = lib_dir.join(profile);

    if profile_dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", profile_dir.display());
    }
}

fn link_system_ggml_libraries(link_kind: &str) {
    if !cfg!(feature = "system-ggml") {
        return;
    }

    println!("cargo:rustc-link-lib={link_kind}=ggml");
    println!("cargo:rustc-link-lib={link_kind}=ggml-base");
    println!("cargo:rustc-link-lib={link_kind}=ggml-cpu");
}

fn link_cuda_libraries(build_shared_libs: bool) {
    if !cfg!(feature = "cuda") || build_shared_libs {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    for lib_dir in find_cuda_helper::find_cuda_lib_dirs() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    if cfg!(target_os = "windows") {
        link_cuda_windows();
    } else {
        link_cuda_unix();
    }
}

fn link_cuda_windows() {
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }
}

fn link_cuda_unix() {
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=cublas_static");
    println!("cargo:rustc-link-lib=static=cublasLt_static");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }

    println!("cargo:rustc-link-lib=static=culibos");
}

fn link_rocm_libraries(build_shared_libs: bool) {
    if !cfg!(feature = "rocm") || build_shared_libs {
        return;
    }

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let rocm_path = env::var("ROCM_PATH")
        .or_else(|_| env::var("HIP_PATH"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:\\Program Files\\AMD\\ROCm".to_string()
            } else {
                "/opt/rocm".to_string()
            }
        });

    let rocm_lib = Path::new(&rocm_path).join("lib");

    assert!(
        rocm_lib.exists(),
        "ROCm libraries not found at: {}\n\
         Please install ROCm or set ROCM_PATH/HIP_PATH environment variable.\n\
         Download from: https://rocm.docs.amd.com/",
        rocm_lib.display()
    );

    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=hipblas");
}

fn link_openmp(target_triple: &str) {
    if cfg!(feature = "openmp") && target_triple.contains("gnu") {
        println!("cargo:rustc-link-lib=gomp");
    }
}

fn link_platform_system_libraries(target_os: &TargetOs) {
    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            link_msvc_system_libraries();
        }
        TargetOs::Linux => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(variant) => {
            link_apple_frameworks(*variant);
        }
        TargetOs::Android => {
            link_android_cpp_stdlib();
        }
        TargetOs::Windows(_) => {}
    }
}

fn link_android_cpp_stdlib() {
    if cfg!(feature = "static-stdcxx") {
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=c++abi");
    } else if cfg!(feature = "shared-stdcxx") {
        println!("cargo:rustc-link-lib=c++_shared");
    }
}

fn link_msvc_system_libraries() {
    println!("cargo:rustc-link-lib=advapi32");

    let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .contains("crt-static");

    if cfg!(debug_assertions) {
        if crt_static {
            println!("cargo:rustc-link-lib=libcmtd");
        } else {
            println!("cargo:rustc-link-lib=dylib=msvcrtd");
        }
    }
}

fn link_apple_frameworks(variant: AppleVariant) {
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");

    if let AppleVariant::MacOS = variant
        && let Some(path) = macos_link_search_path()
    {
        println!("cargo:rustc-link-lib=clang_rt.osx");
        println!("cargo:rustc-link-search={path}");
    }
}

fn macos_link_search_path() -> Option<String> {
    let output = std::process::Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;

    if !output.status.success() {
        println!(
            "cargo:warning=failed to run 'clang --print-search-dirs', continuing without a link search path"
        );

        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;

            return Some(format!("{path}/lib/darwin"));
        }
    }

    println!("cargo:warning=failed to determine link search path, continuing without it");

    None
}
