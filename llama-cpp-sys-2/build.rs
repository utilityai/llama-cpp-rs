use cmake::Config;
use glob::glob;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::DirEntry;

#[cfg(feature = "compat")]
const PREFIX: &str = "llm_";
const CMP0147_OLD_BLOCK: &str = r#"if (POLICY CMP0147)
    # Parallel build custom build steps
    cmake_policy(SET CMP0147 NEW)
endif()"#;
const CMP0147_PATCHED_BLOCK: &str = r#"if (POLICY CMP0147)
    # CMake 3.27+ enables parallel custom build steps for Visual Studio via CMP0147.
    # ggml-vulkan relies on ordered ExternalProject steps; parallelized steps can run
    # out of order under MSBuild (build/install before configure), causing missing CMakeCache.
    if (CMAKE_GENERATOR MATCHES "Visual Studio")
        cmake_policy(SET CMP0147 OLD)
    else()
        cmake_policy(SET CMP0147 NEW)
    endif()
endif()"#;

enum WindowsVariant {
    Msvc,
    Other,
}

enum AppleVariant {
    MacOS,
    Other,
}

enum TargetOs {
    Windows(WindowsVariant),
    Apple(AppleVariant),
    Linux,
    Android,
}

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

fn parse_target_os() -> Result<(TargetOs, String), String> {
    let target = env::var("TARGET").unwrap();

    if target.contains("windows") {
        if target.ends_with("-windows-msvc") {
            Ok((TargetOs::Windows(WindowsVariant::Msvc), target))
        } else {
            Ok((TargetOs::Windows(WindowsVariant::Other), target))
        }
    } else if target.contains("apple") {
        if target.ends_with("-apple-darwin") {
            Ok((TargetOs::Apple(AppleVariant::MacOS), target))
        } else {
            Ok((TargetOs::Apple(AppleVariant::Other), target))
        }
    } else if target.contains("android")
        || target == "aarch64-linux-android"
        || target == "armv7-linux-androideabi"
        || target == "i686-linux-android"
        || target == "x86_64-linux-android"
    {
        // Handle both full android targets and short names like arm64-v8a that cargo ndk might use
        Ok((TargetOs::Android, target))
    } else if target.contains("linux") {
        Ok((TargetOs::Linux, target))
    } else {
        Err(target)
    }
}

fn get_cargo_target_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let path = PathBuf::from(out_dir);
    let target_dir = path
        .ancestors()
        .nth(3)
        .ok_or("OUT_DIR is not deep enough")?;
    Ok(target_dir.to_path_buf())
}

fn extract_lib_names(out_dir: &Path, build_shared_libs: bool) -> Vec<String> {
    let lib_pattern = if cfg!(windows) {
        "*.lib"
    } else if cfg!(target_os = "macos") {
        if build_shared_libs {
            "*.dylib"
        } else {
            "*.a"
        }
    } else if build_shared_libs {
        "*.so"
    } else {
        "*.a"
    };
    let libs_dir = out_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let mut lib_names: Vec<String> = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Remove the "lib" prefix if present
                let lib_name = if stem_str.starts_with("lib") {
                    stem_str.strip_prefix("lib").unwrap_or(stem_str)
                } else {
                    if path.extension() == Some(std::ffi::OsStr::new("a")) {
                        let target = path.parent().unwrap().join(format!("lib{}.a", stem_str));
                        std::fs::rename(&path, &target).unwrap_or_else(|e| {
                            panic!("Failed to rename {path:?} to {target:?}: {e:?}");
                        })
                    }
                    stem_str
                };
                lib_names.push(lib_name.to_string());
            }
            Err(e) => println!("cargo:warning=error={}", e),
        }
    }
    // Sort in reverse dependency order for single-pass linkers.
    // Strip `llm_` prefix so the sort works for both normal and compat builds.
    lib_names.sort_by_key(|name| {
        let base = name.strip_prefix("llm_").unwrap_or(name);
        match base {
            "common" => 0,
            "llama" => 1,
            "ggml" => 2,
            "ggml-cpu" => 3,
            "ggml-metal" => 4,
            "ggml-base" => 5,
            _ => 3,
        }
    });

    lib_names
}

fn extract_lib_assets(out_dir: &Path) -> Vec<PathBuf> {
    let shared_lib_pattern = if cfg!(windows) {
        "*.dll"
    } else if cfg!(target_os = "macos") {
        "*.dylib"
    } else {
        "*.so"
    };

    let shared_libs_dir = if cfg!(windows) { "bin" } else { "lib" };
    let libs_dir = out_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());
    let mut files = Vec::new();

    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                files.push(path);
            }
            Err(e) => eprintln!("cargo:warning=error={}", e),
        }
    }

    files
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}

fn validate_android_ndk(ndk_path: &str) -> Result<(), String> {
    let ndk_path = Path::new(ndk_path);

    if !ndk_path.exists() {
        return Err(format!(
            "Android NDK path does not exist: {}",
            ndk_path.display()
        ));
    }

    let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");
    if !toolchain_file.exists() {
        return Err(format!(
            "Android NDK toolchain file not found: {}\n\
             This indicates an incomplete NDK installation.",
            toolchain_file.display()
        ));
    }

    Ok(())
}

fn is_hidden(e: &DirEntry) -> bool {
    e.file_name()
        .to_str()
        .map(|s| s.starts_with('.'))
        .unwrap_or_default()
}

fn patch_windows_vulkan_cmp0147(manifest_dir: &str) {
    let cmake_path = Path::new(manifest_dir)
        .join("llama.cpp")
        .join("ggml")
        .join("src")
        .join("ggml-vulkan")
        .join("CMakeLists.txt");

    let original = match std::fs::read_to_string(&cmake_path) {
        Ok(content) => content,
        Err(_) => return,
    };

    let had_crlf = original.contains("\r\n");
    let mut normalized = original.replace("\r\n", "\n");

    if normalized.contains(CMP0147_PATCHED_BLOCK) {
        return;
    }
    if !normalized.contains(CMP0147_OLD_BLOCK) {
        return;
    }

    normalized = normalized.replace(CMP0147_OLD_BLOCK, CMP0147_PATCHED_BLOCK);
    let updated = if had_crlf {
        normalized.replace('\n', "\r\n")
    } else {
        normalized
    };

    if let Err(error) = std::fs::write(&cmake_path, updated) {
        println!(
            "cargo:warning=Failed to patch ggml-vulkan CMP0147 policy at {}: {}",
            cmake_path.display(),
            error
        );
    }
}

#[cfg(feature = "compat")]
#[derive(Debug)]
struct GGMLLinkRename;

#[cfg(feature = "compat")]
impl bindgen::callbacks::ParseCallbacks for GGMLLinkRename {
    fn generated_link_name_override(
        &self,
        item_info: bindgen::callbacks::ItemInfo<'_>,
    ) -> Option<String> {
        if matches!(item_info.kind, bindgen::callbacks::ItemKind::Function)
            && item_info.name.starts_with("ggml_")
        {
            Some(format!("{PREFIX}{}", item_info.name))
        } else {
            None
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let (target_os, target_triple) =
        parse_target_os().unwrap_or_else(|t| panic!("Failed to parse target os {t}"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let llama_src = Path::new(&manifest_dir).join("llama.cpp");
    let build_shared_libs = cfg!(feature = "dynamic-link");

    let build_shared_libs = std::env::var("LLAMA_BUILD_SHARED_LIBS")
        .map(|v| v == "1")
        .unwrap_or(build_shared_libs);
    let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or("Release".to_string());
    let static_crt = env::var("LLAMA_STATIC_CRT")
        .map(|v| v == "1")
        .unwrap_or(false);

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC_CRT");

    debug_log!("TARGET: {}", target_triple);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());
    debug_log!("BUILD_SHARED: {}", build_shared_libs);

    // Make sure that changes to the llama.cpp project trigger a rebuild.
    let rebuild_on_children_of = [
        llama_src.join("src"),
        llama_src.join("ggml/src"),
        llama_src.join("common"),
    ];
    for entry in walkdir::WalkDir::new(&llama_src)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        let entry = entry.expect("Failed to obtain entry");
        let rebuild = entry
            .file_name()
            .to_str()
            .map(|f| f.starts_with("CMake"))
            .unwrap_or_default()
            || rebuild_on_children_of
                .iter()
                .any(|src_folder| entry.path().starts_with(src_folder));
        if rebuild {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    // Speed up build
    env::set_var(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        std::thread::available_parallelism()
            .unwrap()
            .get()
            .to_string(),
    );

    // Bindings
    let mut bindings_builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("llama_rs_.*")
        .allowlist_type("llama_rs_.*")
        .prepend_enum_name(false);

    // Configure mtmd feature if enabled
    if cfg!(feature = "mtmd") {
        bindings_builder = bindings_builder
            .header("wrapper_mtmd.h")
            .allowlist_function("mtmd_.*")
            .allowlist_type("mtmd_.*");
    }

    // Configure Android-specific bindgen settings
    if matches!(target_os, TargetOs::Android) {
        // Detect Android NDK from environment variables
        let android_ndk = env::var("ANDROID_NDK")
            .or_else(|_| env::var("ANDROID_NDK_ROOT"))
            .or_else(|_| env::var("NDK_ROOT"))
            .or_else(|_| env::var("CARGO_NDK_ANDROID_NDK"))
            .or_else(|_| {
                // Try to auto-detect NDK from Android SDK
                if let Some(home) = env::home_dir() {
                    let android_home = env::var("ANDROID_HOME")
                        .or_else(|_| env::var("ANDROID_SDK_ROOT"))
                        .unwrap_or_else(|_| format!("{}/Android/Sdk", home.display()));

                    let ndk_dir = format!("{}/ndk", android_home);
                    if let Ok(entries) = std::fs::read_dir(&ndk_dir) {
                        let mut versions: Vec<_> = entries
                            .filter_map(|e| e.ok())
                            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                            .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                            .collect();
                        versions.sort();
                        if let Some(latest) = versions.last() {
                            return Ok(format!("{}/{}", ndk_dir, latest));
                        }
                    }
                }
                Err(env::VarError::NotPresent)
            })
            .unwrap_or_else(|_| {
                panic!(
                    "Android NDK not found. Please set one of: ANDROID_NDK, NDK_ROOT, ANDROID_NDK_ROOT\n\
                     Current target: {}\n\
                     Download from: https://developer.android.com/ndk/downloads",
                    target_triple
                );
            });

        // Get Android API level
        let android_api = env::var("ANDROID_API_LEVEL")
            .or_else(|_| env::var("ANDROID_PLATFORM").map(|p| p.replace("android-", "")))
            .or_else(|_| env::var("CARGO_NDK_ANDROID_PLATFORM").map(|p| p.replace("android-", "")))
            .unwrap_or_else(|_| "28".to_string());

        // Determine host platform
        let host_tag = if cfg!(target_os = "macos") {
            "darwin-x86_64"
        } else if cfg!(target_os = "linux") {
            "linux-x86_64"
        } else if cfg!(target_os = "windows") {
            "windows-x86_64"
        } else {
            panic!("Unsupported host platform for Android NDK");
        };

        // Map Rust target to Android architecture
        let android_target_prefix = if target_triple.contains("aarch64") {
            "aarch64-linux-android"
        } else if target_triple.contains("armv7") {
            "arm-linux-androideabi"
        } else if target_triple.contains("x86_64") {
            "x86_64-linux-android"
        } else if target_triple.contains("i686") {
            "i686-linux-android"
        } else {
            panic!("Unsupported Android target: {}", target_triple);
        };

        // Setup Android toolchain paths
        let toolchain_path = format!("{}/toolchains/llvm/prebuilt/{}", android_ndk, host_tag);
        let sysroot = format!("{}/sysroot", toolchain_path);

        // Validate toolchain existence
        if !std::path::Path::new(&toolchain_path).exists() {
            panic!(
                "Android NDK toolchain not found at: {}\n\
                 Please ensure you have the correct Android NDK for your platform.",
                toolchain_path
            );
        }

        // Find clang builtin includes
        let clang_builtin_includes = {
            let clang_lib_path = format!("{}/lib/clang", toolchain_path);
            std::fs::read_dir(&clang_lib_path).ok().and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|entry| {
                        entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                            && entry
                                .file_name()
                                .to_str()
                                .map(|name| name.chars().next().unwrap_or('0').is_ascii_digit())
                                .unwrap_or(false)
                    })
                    .and_then(|entry| {
                        let include_path =
                            format!("{}/{}/include", clang_lib_path, entry.file_name().to_str()?);
                        if std::path::Path::new(&include_path).exists() {
                            Some(include_path)
                        } else {
                            None
                        }
                    })
            })
        };

        // Configure bindgen for Android
        bindings_builder = bindings_builder
            .clang_arg(format!("--sysroot={}", sysroot))
            .clang_arg(format!("-D__ANDROID_API__={}", android_api))
            .clang_arg("-D__ANDROID__");

        // Add include paths in correct order
        if let Some(ref builtin_includes) = clang_builtin_includes {
            bindings_builder = bindings_builder
                .clang_arg("-isystem")
                .clang_arg(builtin_includes);
        }

        bindings_builder = bindings_builder
            .clang_arg("-isystem")
            .clang_arg(format!("{}/usr/include/{}", sysroot, android_target_prefix))
            .clang_arg("-isystem")
            .clang_arg(format!("{}/usr/include", sysroot))
            .clang_arg("-include")
            .clang_arg("stdbool.h")
            .clang_arg("-include")
            .clang_arg("stdint.h");

        // Set additional clang args for cargo ndk compatibility
        if env::var("CARGO_SUBCOMMAND").as_deref() == Ok("ndk") {
            std::env::set_var(
                "BINDGEN_EXTRA_CLANG_ARGS",
                format!("--target={}", target_triple),
            );
        }
    }

    // Fix bindgen header discovery on Windows MSVC
    // Use cc crate to discover MSVC include paths by compiling a dummy file
    if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
        // Create a minimal dummy C file to extract compiler flags
        let out_dir = env::var("OUT_DIR").unwrap();
        let dummy_c = Path::new(&out_dir).join("dummy.c");
        std::fs::write(&dummy_c, "int main() { return 0; }").unwrap();

        // Use cc crate to get compiler with proper environment setup
        let mut build = cc::Build::new();
        build.file(&dummy_c);

        // Get the actual compiler command cc would use
        let compiler = build.try_get_compiler().unwrap();

        // Extract include paths by checking compiler's environment
        // cc crate sets up MSVC environment internally
        let env_include = compiler
            .env()
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("INCLUDE"))
            .map(|(_, v)| v);

        if let Some(include_paths) = env_include {
            for include_path in include_paths
                .to_string_lossy()
                .split(';')
                .filter(|s| !s.is_empty())
            {
                bindings_builder = bindings_builder
                    .clang_arg("-isystem")
                    .clang_arg(include_path);
                debug_log!("Added MSVC include path: {}", include_path);
            }
        }

        // Add MSVC compatibility flags
        bindings_builder = bindings_builder
            .clang_arg(format!("--target={}", target_triple))
            .clang_arg("-fms-compatibility")
            .clang_arg("-fms-extensions");

        debug_log!(
            "Configured bindgen with MSVC toolchain for target: {}",
            target_triple
        );
    }
    #[cfg(feature = "compat")]
    {
        bindings_builder = bindings_builder.parse_callbacks(Box::new(GGMLLinkRename));
    }

    let bindings = bindings_builder
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper_common.h");
    println!("cargo:rerun-if-changed=wrapper_common.cpp");
    println!("cargo:rerun-if-changed=wrapper_oai.h");
    println!("cargo:rerun-if-changed=wrapper_oai.cpp");
    println!("cargo:rerun-if-changed=wrapper_utils.h");
    println!("cargo:rerun-if-changed=wrapper_mtmd.h");

    debug_log!("Bindings Created");

    let mut common_wrapper_build = cc::Build::new();
    common_wrapper_build
        .cpp(true)
        .file("wrapper_common.cpp")
        .file("wrapper_oai.cpp")
        .include(&llama_src)
        .include(llama_src.join("common"))
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);
    if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
        common_wrapper_build.flag("/std:c++17");
    }
    common_wrapper_build.compile("llama_cpp_sys_2_common_wrapper");

    // Build with Cmake

    let mut config = Config::new(&llama_src);

    // Would require extra source files to pointlessly
    // be included in what's uploaded to and downloaded from
    // crates.io, so deactivating these instead
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_CURL", "OFF");

    if cfg!(feature = "mtmd") {
        // mtmd support in llama-cpp is within the tools directory
        config.define("LLAMA_BUILD_TOOLS", "ON");
    }

    // Pass CMAKE_ environment variables down to CMake
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }

    // extract the target-cpu config value, if specified
    let target_cpu = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .and_then(|rustflags| {
            rustflags
                .split('\x1f')
                .find(|f| f.contains("target-cpu="))
                .and_then(|f| f.split("target-cpu=").nth(1))
                .map(|s| s.to_string())
        });

    if target_cpu == Some("native".into()) {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");
    }
    // if native isn't specified, enable specific features for ggml instead
    else {
        // rust code isn't using `target-cpu=native`, so llama.cpp shouldn't use GGML_NATIVE either
        config.define("GGML_NATIVE", "OFF");

        // if `target-cpu` is set set, also set -march for llama.cpp to the same value
        if let Some(ref cpu) = target_cpu {
            debug_log!("Setting baseline architecture: -march={}", cpu);
            config.cflag(&format!("-march={}", cpu));
            config.cxxflag(&format!("-march={}", cpu));
        }

        // I expect this env var to always be present
        let features = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .expect("Env var CARGO_CFG_TARGET_FEATURE not found.");
        debug_log!("Compiling with target features: {}", features);

        // list of rust target_features here:
        //   https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute
        // GGML config flags have been found by looking at:
        //   llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt
        for feature in features.split(',') {
            match feature {
                "avx" => {
                    config.define("GGML_AVX", "ON");
                }
                "avx2" => {
                    config.define("GGML_AVX2", "ON");
                }
                "avx512bf16" => {
                    config.define("GGML_AVX512_BF16", "ON");
                }
                "avx512vbmi" => {
                    config.define("GGML_AVX512_VBMI", "ON");
                }
                "avx512vnni" => {
                    config.define("GGML_AVX512_VNNI", "ON");
                }
                "avxvnni" => {
                    config.define("GGML_AVX_VNNI", "ON");
                }
                "bmi2" => {
                    config.define("GGML_BMI2", "ON");
                }
                "f16c" => {
                    config.define("GGML_F16C", "ON");
                }
                "fma" => {
                    config.define("GGML_FMA", "ON");
                }
                "sse4.2" => {
                    config.define("GGML_SSE42", "ON");
                }
                _ => {
                    debug_log!(
                        "Unrecognized cpu feature: '{}' - skipping GGML config for it.",
                        feature
                    );
                    continue;
                }
            };
        }
    }

    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );

    if matches!(target_os, TargetOs::Apple(_)) {
        config.define("GGML_BLAS", "OFF");
    }

    if (matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc))
        && matches!(
            profile.as_str(),
            "Release" | "RelWithDebInfo" | "MinSizeRel"
        ))
    {
        // Debug Rust builds under MSVC turn off optimization even though we're ideally building the release profile of llama.cpp.
        // Looks like an upstream bug:
        // https://github.com/rust-lang/cmake-rs/issues/240
        // For now explicitly reinject the optimization flags that a CMake Release build is expected to have on in this scenario.
        // This fixes CPU inference performance when part of a Rust debug build.
        for flag in &["/O2", "/DNDEBUG", "/Ob2"] {
            config.cflag(flag);
            config.cxxflag(flag);
        }
    }

    config.static_crt(static_crt);

    if matches!(target_os, TargetOs::Android) {
        // Android NDK Build Configuration
        let android_ndk = env::var("ANDROID_NDK")
            .or_else(|_| env::var("NDK_ROOT"))
            .or_else(|_| env::var("ANDROID_NDK_ROOT"))
            .unwrap_or_else(|_| {
                panic!(
                    "Android NDK not found. Please set one of: ANDROID_NDK, NDK_ROOT, ANDROID_NDK_ROOT\n\
                     Download from: https://developer.android.com/ndk/downloads"
                );
            });

        // Validate NDK installation
        if let Err(error) = validate_android_ndk(&android_ndk) {
            panic!("Android NDK validation failed: {}", error);
        }

        // Rerun build script if NDK environment variables change
        println!("cargo:rerun-if-env-changed=ANDROID_NDK");
        println!("cargo:rerun-if-env-changed=NDK_ROOT");
        println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");

        // Set CMake toolchain file for Android
        let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", android_ndk);
        config.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);

        // Configure Android platform (API level)
        let android_platform = env::var("ANDROID_PLATFORM").unwrap_or_else(|_| {
            env::var("ANDROID_API_LEVEL")
                .map(|level| format!("android-{}", level))
                .unwrap_or_else(|_| "android-28".to_string())
        });

        println!("cargo:rerun-if-env-changed=ANDROID_PLATFORM");
        println!("cargo:rerun-if-env-changed=ANDROID_API_LEVEL");
        config.define("ANDROID_PLATFORM", &android_platform);

        // Map Rust target to Android ABI
        let android_abi = if target_triple.contains("aarch64") {
            "arm64-v8a"
        } else if target_triple.contains("armv7") {
            "armeabi-v7a"
        } else if target_triple.contains("x86_64") {
            "x86_64"
        } else if target_triple.contains("i686") {
            "x86"
        } else {
            panic!(
                "Unsupported Android target: {}\n\
                 Supported targets: aarch64-linux-android, armv7-linux-androideabi, i686-linux-android, x86_64-linux-android",
                target_triple
            );
        };

        config.define("ANDROID_ABI", android_abi);

        // Configure architecture-specific compiler flags
        match android_abi {
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

        // Android-specific CMake configurations
        config.define("GGML_LLAMAFILE", "OFF");

        // Link Android system libraries
        println!("cargo:rustc-link-lib=log");
        println!("cargo:rustc-link-lib=android");
    }

    if matches!(target_os, TargetOs::Linux)
        && target_triple.contains("aarch64")
        && target_cpu != Some("native".into())
    {
        // If the target-cpu is not specified as native, we take off the native ARM64 support.
        // It is useful in docker environments where the native feature is not enabled.
        config.define("GGML_NATIVE", "OFF");
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }

    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        match target_os {
            TargetOs::Windows(_) => {
                // Keep Visual Studio ExternalProject steps ordered; NEW breaks ggml-vulkan on Windows.
                patch_windows_vulkan_cmp0147(&manifest_dir);

                let vulkan_path = env::var("VULKAN_SDK").expect(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set",
                );
                let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
                println!("cargo:rustc-link-lib=vulkan-1");

                // workaround for this error: "FileTracker : error FTK1011: could not create the new file tracking log file"
                // it has to do with MSBuild FileTracker not respecting the path
                // limit configuration set in the windows registry.
                // I'm not sure why that's a thing, but this makes my builds work.
                // (crates that depend on llama-cpp-rs w/ vulkan easily exceed the default PATH_MAX on windows)
                env::set_var("TrackFileAccess", "false");
                // since we disabled TrackFileAccess, we can now run into problems with parallel
                // access to pdb files. /FS solves this.
                config.cflag("/FS");
                config.cxxflag("/FS");
            }
            TargetOs::Linux => {
                // If we are not using system provided vulkan SDK, add vulkan libs for linking
                if let Ok(vulkan_path) = env::var("VULKAN_SDK") {
                    let vulkan_lib_path = Path::new(&vulkan_path).join("lib");
                    println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
                }
                println!("cargo:rustc-link-lib=vulkan");
            }
            _ => (),
        }
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");

        if cfg!(feature = "cuda-no-vmm") {
            config.define("GGML_CUDA_NO_VMM", "ON");
        }
    }

    // Android doesn't have OpenMP support AFAICT and openmp is a default feature. Do this here
    // rather than modifying the defaults in Cargo.toml just in case someone enables the OpenMP feature
    // and tries to build for Android anyway.
    if cfg!(feature = "openmp") && !matches!(target_os, TargetOs::Android) {
        config.define("GGML_OPENMP", "ON");
    } else {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(feature = "system-ggml") {
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    let build_dir = config.build();

    #[cfg(feature = "compat")]
    {
        compat::redefine_symbols(&out_dir);
    }

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    if cfg!(feature = "system-ggml") {
        // Extract library directory from CMake's found GGML package
        let cmake_cache = build_dir.join("build").join("CMakeCache.txt");
        if let Ok(cache_contents) = std::fs::read_to_string(&cmake_cache) {
            let mut ggml_lib_dirs = std::collections::HashSet::new();

            // Parse CMakeCache.txt to find where GGML libraries were found
            for line in cache_contents.lines() {
                if line.starts_with("GGML_LIBRARY:")
                    || line.starts_with("GGML_BASE_LIBRARY:")
                    || line.starts_with("GGML_CPU_LIBRARY:")
                {
                    if let Some(lib_path) = line.split('=').nth(1) {
                        if let Some(parent) = Path::new(lib_path).parent() {
                            ggml_lib_dirs.insert(parent.to_path_buf());
                        }
                    }
                }
            }

            // Add each unique library directory to the search path
            for lib_dir in ggml_lib_dirs {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                debug_log!("Added system GGML library path: {}", lib_dir.display());
            }
        }
    }

    if cfg!(feature = "cuda") && !build_shared_libs {
        // Re-run build script if CUDA_PATH environment variable changes
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        // Add CUDA library directories to the linker search path
        for lib_dir in find_cuda_helper::find_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }

        // Platform-specific linking
        if cfg!(target_os = "windows") {
            // ✅ On Windows, use dynamic linking.
            // Static linking is problematic because NVIDIA does not provide culibos.lib,
            // and static CUDA libraries (like cublas_static.lib) are usually not shipped.

            println!("cargo:rustc-link-lib=cudart"); // Links to cudart64_*.dll
            println!("cargo:rustc-link-lib=cublas"); // Links to cublas64_*.dll
            println!("cargo:rustc-link-lib=cublasLt"); // Links to cublasLt64_*.dll

            // Link to CUDA driver API (nvcuda.dll via cuda.lib)
            if !cfg!(feature = "cuda-no-vmm") {
                println!("cargo:rustc-link-lib=cuda");
            }
        } else {
            // ✅ On non-Windows platforms (e.g., Linux), static linking is preferred and supported.
            // Static libraries like cudart_static and cublas_static depend on culibos.

            println!("cargo:rustc-link-lib=static=cudart_static");
            println!("cargo:rustc-link-lib=static=cublas_static");
            println!("cargo:rustc-link-lib=static=cublasLt_static");

            // Link to CUDA driver API (libcuda.so)
            if !cfg!(feature = "cuda-no-vmm") {
                println!("cargo:rustc-link-lib=cuda");
            }

            // culibos is required when statically linking cudart_static
            println!("cargo:rustc-link-lib=static=culibos");
        }
    }

    // Link libraries
    let llama_libs_kind = if build_shared_libs || cfg!(feature = "system-ggml") {
        "dylib"
    } else {
        "static"
    };
    let llama_libs = extract_lib_names(&out_dir, build_shared_libs);
    assert_ne!(llama_libs.len(), 0);

    let common_lib_dir = out_dir.join("build").join("common");
    if common_lib_dir.is_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            common_lib_dir.display()
        );
        let common_profile_dir = common_lib_dir.join(&profile);
        if common_profile_dir.is_dir() {
            println!(
                "cargo:rustc-link-search=native={}",
                common_profile_dir.display()
            );
        }
        if cfg!(feature = "compat") && cfg!(target_os = "windows") {
            println!("cargo:rustc-link-lib=static=llm_common");
        } else {
            println!("cargo:rustc-link-lib=static=common");
        }
    }

    if cfg!(feature = "system-ggml") {
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml");
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml-base");
        println!("cargo:rustc-link-lib={llama_libs_kind}=ggml-cpu");
    }
    for lib in llama_libs {
        let link = format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib);
        debug_log!("LINK {link}",);
        println!("{link}",);
    }

    // OpenMP
    if cfg!(feature = "openmp") && target_triple.contains("gnu") {
        println!("cargo:rustc-link-lib=gomp");
    }

    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            println!("cargo:rustc-link-lib=advapi32");
            if cfg!(debug_assertions) {
                println!("cargo:rustc-link-lib=dylib=msvcrtd");
            }
        }
        TargetOs::Linux => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(variant) => {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=c++");

            match variant {
                AppleVariant::MacOS => {
                    // On (older) OSX we need to link against the clang runtime,
                    // which is hidden in some non-default path.
                    //
                    // More details at https://github.com/alexcrichton/curl-rust/issues/279.
                    if let Some(path) = macos_link_search_path() {
                        println!("cargo:rustc-link-lib=clang_rt.osx");
                        println!("cargo:rustc-link-search={}", path);
                    }
                }
                AppleVariant::Other => (),
            }
        }
        _ => (),
    }

    // copy DLLs to target
    if build_shared_libs {
        let libs_assets = extract_lib_assets(&out_dir);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    std::fs::hard_link(asset.clone(), dst).unwrap();
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }
        }
    }
}

/// Prefix ggml/gguf symbols in static libraries to avoid collisions with other
/// ggml-bundling crates (e.g. whisper-rs-sys). Requires nm + objcopy (or LLVM equivalents).
#[cfg(feature = "compat")]
mod compat {
    use std::collections::HashSet;
    use std::fmt::{Display, Formatter};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use crate::PREFIX;

    const MACHO_UNDERSCORE: bool =
        cfg!(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"));

    pub fn redefine_symbols(out_dir: &Path) {
        let (nm, objcopy) = tools();

        let mut libs = Vec::new();
        let mut scan_dirs = vec![out_dir.join("lib"), out_dir.join("lib64")];
        if cfg!(target_os = "windows") {
            scan_dirs.extend([
                out_dir.join("build").join("common"),
                out_dir.join("build").join("common").join("Release"),
                out_dir.join("build").join("common").join("RelWithDebInfo"),
                out_dir.join("build").join("common").join("MinSizeRel"),
                out_dir.join("build").join("common").join("Debug"),
            ]);
        }

        for dir in scan_dirs {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if is_static_lib(&path) {
                        libs.push(path);
                    }
                }
            }
        }

        if libs.is_empty() {
            println!("cargo:warning=compat: no static libraries found, skipping symbol rewrite");
            return;
        }

        let sym_prefix = if MACHO_UNDERSCORE { "_" } else { "" };

        let filters = [
            Filter { prefix: format!("{sym_prefix}ggml_"), sym_types: &['T', 'U', 'B', 'D', 'S'] },
            Filter { prefix: format!("{sym_prefix}gguf_"), sym_types: &['T', 'U', 'B', 'D', 'S'] },
            Filter { prefix: format!("{sym_prefix}quantize_"), sym_types: &['T', 'U'] },
            Filter { prefix: format!("{sym_prefix}dequantize_"), sym_types: &['T', 'U'] },
            Filter { prefix: format!("{sym_prefix}iq2xs_"), sym_types: &['T', 'U'] },
            Filter { prefix: format!("{sym_prefix}iq3xs_"), sym_types: &['T', 'U'] },
        ];

        let cpp_mangled_prefix = format!("{sym_prefix}_Z");

        for lib_path in &libs {
            let lib_name = lib_path.file_name().unwrap().to_str().unwrap();
            let lib_dir = lib_path.parent().unwrap();

            let nm_output = nm_symbols(&nm, lib_name, lib_dir);
            let c_symbols = get_symbols(&nm_output, &filters);

            let cpp_symbols = get_cpp_ggml_symbols(&nm_output, &cpp_mangled_prefix);

            if !c_symbols.is_empty() || !cpp_symbols.is_empty() {
                eprintln!(
                    "compat: rewriting {} C + {} C++ symbols in {}",
                    c_symbols.len(),
                    cpp_symbols.len(),
                    lib_name
                );
                objcopy_rewrite(&objcopy, lib_name, PREFIX, &c_symbols, &cpp_symbols, lib_dir);
            }
        }

        // Rename libraries (libX.a → libllm_X.a) to avoid ambiguous -l flags.
        for lib_path in &libs {
            let file_name = lib_path.file_name().unwrap().to_str().unwrap();
            let lib_dir = lib_path.parent().unwrap();

            let new_name = if let Some(rest) = file_name.strip_prefix("lib") {
                format!("libllm_{rest}")
            } else {
                format!("llm_{file_name}")
            };

            let new_path = lib_dir.join(&new_name);
            std::fs::rename(lib_path, &new_path).unwrap_or_else(|e| {
                panic!("compat: failed to rename {file_name} → {new_name}: {e}");
            });
            eprintln!("compat: renamed {file_name} → {new_name}");
        }
    }

    fn is_static_lib(path: &Path) -> bool {
        match path.extension().and_then(|e| e.to_str()) {
            Some("a") => true,
            Some("lib") if cfg!(target_family = "windows") => true,
            _ => false,
        }
    }

    enum Tool {
        Name(&'static str),
        FullPath(PathBuf),
    }

    impl Display for Tool {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                Tool::Name(n) => write!(f, "{n}"),
                Tool::FullPath(p) => write!(f, "{}", p.display()),
            }
        }
    }

    fn tools() -> (Tool, Tool) {
        let (nm_names, nm_help): (Vec<&str>, Vec<&str>) = if cfg!(target_os = "linux") {
            (vec!["nm", "llvm-nm"], vec!["GNU nm", "llvm-nm"])
        } else {
            (vec!["nm", "llvm-nm"], vec!["nm (Xcode CLT)", "llvm-nm"])
        };

        let (objcopy_names, objcopy_help): (Vec<&str>, Vec<&str>) = if cfg!(target_os = "linux") {
            (
                vec!["objcopy", "llvm-objcopy"],
                vec!["GNU objcopy", "llvm-objcopy"],
            )
        } else {
            (vec!["llvm-objcopy"], vec!["llvm-objcopy (brew install llvm)"])
        };

        println!("cargo:rerun-if-env-changed=NM_PATH");
        println!("cargo:rerun-if-env-changed=OBJCOPY_PATH");

        let nm = find_tool(&nm_names, "NM_PATH").unwrap_or_else(|| {
            panic!(
                "compat: no nm-equivalent found in PATH. Install one of: {:?}\n\
                 Or set NM_PATH to its full path.",
                nm_help
            );
        });

        let objcopy = find_tool(&objcopy_names, "OBJCOPY_PATH").unwrap_or_else(|| {
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                panic!(
                    "compat: llvm-objcopy not found. Required for the `compat` feature on macOS.\n\
                     Install via: brew install llvm\n\
                     Or set OBJCOPY_PATH to its full path (e.g. /opt/homebrew/opt/llvm/bin/llvm-objcopy)."
                );
            } else {
                panic!(
                    "compat: no objcopy-equivalent found in PATH. Install one of: {:?}\n\
                     Or set OBJCOPY_PATH to its full path.",
                    objcopy_help
                );
            }
        });

        (nm, objcopy)
    }

    fn find_tool(names: &[&'static str], env_var: &str) -> Option<Tool> {
        if let Ok(path_str) = std::env::var(env_var) {
            let path_str = path_str.trim_matches([' ', '"', '\''].as_slice());
            let path = PathBuf::from(path_str);
            if path.is_file() {
                return Some(Tool::FullPath(path));
            }
            println!("cargo:warning=compat: {env_var}={path_str} is not a valid file, searching PATH");
        }

        for name in names {
            if let Ok(output) = Command::new(name).arg("--version").output() {
                if output.status.success() {
                    return Some(Tool::Name(name));
                }
            }
        }

        // Try llvm-config --prefix to locate LLVM tools
        if let Ok(output) = Command::new("llvm-config").arg("--prefix").output() {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let bin_dir = PathBuf::from(&prefix).join("bin");
                for name in names {
                    let full_path = bin_dir.join(name);
                    if full_path.is_file() {
                        return Some(Tool::FullPath(full_path));
                    }
                }
            }
        }

        if cfg!(any(target_os = "macos", target_os = "ios")) {
            for homebrew_path in [
                "/opt/homebrew/opt/llvm/bin",
                "/usr/local/opt/llvm/bin",
            ] {
                for name in names {
                    let full_path = PathBuf::from(homebrew_path).join(name);
                    if full_path.is_file() {
                        return Some(Tool::FullPath(full_path));
                    }
                }
            }
        }

        None
    }

    fn nm_symbols(tool: &Tool, lib_name: &str, lib_dir: &Path) -> String {
        let output = Command::new(tool.to_string())
            .current_dir(lib_dir)
            .arg(lib_name)
            .args(["-p", "-P"])
            .output()
            .unwrap_or_else(|e| panic!("compat: failed to run \"{tool}\" on {lib_name}: {e}"));

        if !output.status.success() {
            panic!(
                "compat: nm failed on {lib_name} ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        String::from_utf8_lossy(&output.stdout).to_string()
    }

    struct Filter {
        prefix: String,
        sym_types: &'static [char],
    }

    fn parse_nm_line(line: &str) -> Option<(&str, char)> {
        let mut stripped = line;
        while stripped.split(' ').count() > 2 {
            if let Some(idx) = stripped.rfind(' ') {
                stripped = &stripped[..idx];
            } else {
                break;
            }
        }
        let parts: Vec<&str> = stripped.splitn(2, ' ').collect();
        if parts.len() != 2 || parts[1].len() != 1 {
            return None;
        }
        Some((parts[0], parts[1].chars().next()?))
    }

    fn get_symbols<'a>(nm_output: &'a str, filters: &[Filter]) -> HashSet<&'a str> {
        nm_output
            .lines()
            .filter_map(|line| {
                let (sym_name, sym_type) = parse_nm_line(line)?;
                for filter in filters {
                    if sym_name.starts_with(&filter.prefix)
                        && filter.sym_types.contains(&sym_type)
                    {
                        return Some(sym_name);
                    }
                }
                None
            })
            .collect()
    }

    /// Collect C++ mangled ggml/gguf symbols (Itanium ABI) for localization.
    /// Matches both defined (T, S, D, B) and undefined (U) symbols so that
    /// cross-object references are rewritten consistently.
    fn get_cpp_ggml_symbols<'a>(nm_output: &'a str, cpp_prefix: &str) -> HashSet<&'a str> {
        nm_output
            .lines()
            .filter_map(|line| {
                let (sym_name, sym_type) = parse_nm_line(line)?;
                if !['T', 'S', 'D', 'B', 'U'].contains(&sym_type) {
                    return None;
                }
                let after = sym_name.strip_prefix(cpp_prefix)?;
                if is_ggml_mangled(after) {
                    Some(sym_name)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Match ggml/gguf-prefixed C++ symbols for renaming.
    fn is_ggml_mangled(after_z: &str) -> bool {
        let base = if after_z.starts_with("TV")
            || after_z.starts_with("TI")
            || after_z.starts_with("TS")
        {
            &after_z[2..]
        } else {
            after_z
        };

        if base.starts_with('N') {
            let rest = &base[1..];
            let digits_end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            if digits_end > 0 {
                let after_digits = &rest[digits_end..];
                if after_digits.starts_with("ggml") || after_digits.starts_with("gguf") {
                    return true;
                }
            }
        }

        let digits_end = base
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(base.len());
        if digits_end > 0 {
            let after_digits = &base[digits_end..];
            if after_digits.starts_with("ggml_") || after_digits.starts_with("gguf_") {
                return true;
            }
        }

        false
    }

    fn objcopy_rewrite(
        tool: &Tool,
        lib_name: &str,
        prefix: &str,
        c_symbols: &HashSet<&str>,
        cpp_symbols: &HashSet<&str>,
        lib_dir: &Path,
    ) {
        let mut cmd = Command::new(tool.to_string());
        cmd.current_dir(lib_dir);

        for sym in c_symbols {
            let new_name = if MACHO_UNDERSCORE && sym.starts_with('_') {
                format!("_{prefix}{}", &sym[1..])
            } else {
                format!("{prefix}{sym}")
            };
            cmd.arg(format!("--redefine-sym={sym}={new_name}"));
        }

        for sym in cpp_symbols {
            // Redefine C++ symbols with prefix (same as C symbols) so that
            // cross-object references stay consistent.
            let new_name = if MACHO_UNDERSCORE && sym.starts_with('_') {
                format!("_{prefix}{}", &sym[1..])
            } else {
                format!("{prefix}{sym}")
            };
            cmd.arg(format!("--redefine-sym={sym}={new_name}"));
        }

        cmd.arg(lib_name);

        let output = cmd
            .output()
            .unwrap_or_else(|e| panic!("compat: failed to run \"{tool}\" on {lib_name}: {e}"));

        if !output.status.success() {
            panic!(
                "compat: objcopy failed on {lib_name} ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
