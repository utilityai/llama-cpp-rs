use std::env;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

const CMAKE_AFFECTING_FEATURES: &[(&str, bool)] = &[
    ("cuda", cfg!(feature = "cuda")),
    ("cuda-no-vmm", cfg!(feature = "cuda-no-vmm")),
    ("metal", cfg!(feature = "metal")),
    ("vulkan", cfg!(feature = "vulkan")),
    ("rocm", cfg!(feature = "rocm")),
    ("openmp", cfg!(feature = "openmp")),
    ("dynamic-link", cfg!(feature = "dynamic-link")),
    ("dynamic-backends", cfg!(feature = "dynamic-backends")),
    ("system-ggml", cfg!(feature = "system-ggml")),
    ("system-ggml-static", cfg!(feature = "system-ggml-static")),
    ("shared-stdcxx", cfg!(feature = "shared-stdcxx")),
    ("static-stdcxx", cfg!(feature = "static-stdcxx")),
];

pub fn stable_cmake_build_dir(
    target_dir: &Path,
    target_triple: &str,
    profile: &str,
    static_crt: bool,
    build_shared_libs: bool,
) -> PathBuf {
    if let Ok(override_path) = env::var("LLAMA_CMAKE_BUILD_DIR_OVERRIDE") {
        let path = PathBuf::from(override_path);
        std::fs::create_dir_all(&path).expect("failed to create cmake build directory override");

        return path;
    }

    let mut hasher = DefaultHasher::new();
    target_triple.hash(&mut hasher);
    profile.hash(&mut hasher);
    static_crt.hash(&mut hasher);
    build_shared_libs.hash(&mut hasher);

    for (name, enabled) in CMAKE_AFFECTING_FEATURES {
        name.hash(&mut hasher);
        enabled.hash(&mut hasher);
    }

    let digest = format!("{:016x}", hasher.finish());
    let path = target_dir.join("llama-cpp-cmake-build").join(digest);

    std::fs::create_dir_all(&path).expect("failed to create cmake build directory");

    path
}
