use std::env;
use std::path::{Path, PathBuf};

/// Consolidated Android NDK configuration, computed once and shared between
/// bindgen and `CMake` configuration steps.
#[derive(Debug)]
pub struct AndroidNdk {
    pub ndk_path: String,
    pub api_level: String,
    pub abi: &'static str,
    pub host_tag: &'static str,
    pub toolchain_path: String,
    pub sysroot: String,
    pub target_prefix: &'static str,
    pub clang_builtin_includes: Option<String>,
}

impl AndroidNdk {
    pub fn detect(target_triple: &str) -> Result<Self, String> {
        let ndk_path = detect_ndk_path(target_triple)?;

        validate_ndk_installation(&ndk_path)?;

        let api_level = detect_api_level();
        let abi = target_triple_to_abi(target_triple)?;
        let host_tag = detect_host_tag()?;
        let target_prefix = target_triple_to_ndk_prefix(target_triple)?;
        let toolchain_path = format!("{ndk_path}/toolchains/llvm/prebuilt/{host_tag}");

        if !Path::new(&toolchain_path).exists() {
            return Err(format!(
                "Android NDK toolchain not found at: {toolchain_path}\n\
                 Please ensure you have the correct Android NDK for your platform."
            ));
        }

        let sysroot = format!("{toolchain_path}/sysroot");
        let clang_builtin_includes = find_clang_builtin_includes(&toolchain_path);

        Ok(Self {
            ndk_path,
            api_level,
            abi,
            host_tag,
            toolchain_path,
            sysroot,
            target_prefix,
            clang_builtin_includes,
        })
    }

    pub fn android_platform(&self) -> String {
        format!("android-{}", self.api_level)
    }

    pub fn cmake_toolchain_file(&self) -> String {
        format!("{}/build/cmake/android.toolchain.cmake", self.ndk_path)
    }
}

fn detect_ndk_path(target_triple: &str) -> Result<String, String> {
    env::var("ANDROID_NDK")
        .or_else(|_| env::var("ANDROID_NDK_ROOT"))
        .or_else(|_| env::var("NDK_ROOT"))
        .or_else(|_| env::var("CARGO_NDK_ANDROID_NDK"))
        .or_else(|_| detect_ndk_from_sdk())
        .map_err(|_| {
            format!(
                "Android NDK not found. Please set one of: ANDROID_NDK, NDK_ROOT, ANDROID_NDK_ROOT\n\
                 Current target: {target_triple}\n\
                 Download from: https://developer.android.com/ndk/downloads"
            )
        })
}

fn detect_ndk_from_sdk() -> Result<String, env::VarError> {
    let home = env::home_dir().ok_or(env::VarError::NotPresent)?;

    let android_home = match env::var("ANDROID_HOME")
        .or_else(|_android_home_unset| env::var("ANDROID_SDK_ROOT"))
    {
        Ok(value) => value,
        Err(_neither_env_var_set) => format!("{}/Android/Sdk", home.display()),
    };

    let ndk_dir = format!("{android_home}/ndk");
    let entries =
        std::fs::read_dir(&ndk_dir).map_err(|_directory_unreadable| env::VarError::NotPresent)?;

    let mut versions: Vec<String> = entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_dir()))
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .map(std::string::ToString::to_string)
        })
        .collect();

    versions.sort();

    versions
        .last()
        .map(|latest| format!("{ndk_dir}/{latest}"))
        .ok_or(env::VarError::NotPresent)
}

fn validate_ndk_installation(ndk_path: &str) -> Result<(), String> {
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

fn detect_api_level() -> String {
    env::var("ANDROID_API_LEVEL")
        .or_else(|_| env::var("ANDROID_PLATFORM").map(|platform| platform.replace("android-", "")))
        .or_else(|_| {
            env::var("CARGO_NDK_ANDROID_PLATFORM").map(|platform| platform.replace("android-", ""))
        })
        .unwrap_or_else(|_| "28".to_string())
}

fn detect_host_tag() -> Result<&'static str, String> {
    if cfg!(target_os = "macos") {
        Ok("darwin-x86_64")
    } else if cfg!(target_os = "linux") {
        Ok("linux-x86_64")
    } else if cfg!(target_os = "windows") {
        Ok("windows-x86_64")
    } else {
        Err("Unsupported host platform for Android NDK".to_string())
    }
}

fn target_triple_to_abi(target_triple: &str) -> Result<&'static str, String> {
    if target_triple.contains("aarch64") {
        Ok("arm64-v8a")
    } else if target_triple.contains("armv7") {
        Ok("armeabi-v7a")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64")
    } else if target_triple.contains("i686") {
        Ok("x86")
    } else {
        Err(format!(
            "Unsupported Android target: {target_triple}\n\
             Supported targets: aarch64-linux-android, armv7-linux-androideabi, i686-linux-android, x86_64-linux-android"
        ))
    }
}

fn target_triple_to_ndk_prefix(target_triple: &str) -> Result<&'static str, String> {
    if target_triple.contains("aarch64") {
        Ok("aarch64-linux-android")
    } else if target_triple.contains("armv7") {
        Ok("arm-linux-androideabi")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64-linux-android")
    } else if target_triple.contains("i686") {
        Ok("i686-linux-android")
    } else {
        Err(format!("Unsupported Android target: {target_triple}"))
    }
}

fn find_clang_builtin_includes(toolchain_path: &str) -> Option<String> {
    let clang_lib_path = format!("{toolchain_path}/lib/clang");
    let entries = std::fs::read_dir(&clang_lib_path).ok()?;

    let version_dir = entries.filter_map(std::result::Result::ok).find(|entry| {
        entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(|ch: char| ch.is_ascii_digit()))
    })?;

    let include_path = PathBuf::from(&clang_lib_path)
        .join(version_dir.file_name())
        .join("include");

    if include_path.exists() {
        Some(include_path.to_string_lossy().to_string())
    } else {
        None
    }
}
