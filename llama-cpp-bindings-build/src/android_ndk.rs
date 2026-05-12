use std::env;
use std::path::{Path, PathBuf};

use thiserror::Error;

const DEFAULT_ANDROID_API_LEVEL: &str = "28";

#[derive(Debug, Error)]
pub enum AndroidNdkDetectionError {
    #[error(
        "Android NDK not found for target {target_triple}. Set ANDROID_NDK, ANDROID_NDK_ROOT, NDK_ROOT, or CARGO_NDK_ANDROID_NDK."
    )]
    NdkRootNotConfigured {
        target_triple: String,
        #[source]
        source: env::VarError,
    },
    #[error("Android NDK path does not exist: {path}")]
    NdkRootMissing { path: PathBuf },
    #[error("Android NDK toolchain file not found: {path}")]
    NdkToolchainFileMissing { path: PathBuf },
    #[error("Android NDK toolchain not found at: {path}")]
    NdkToolchainDirectoryMissing { path: PathBuf },
    #[error("Unsupported host platform for Android NDK")]
    UnsupportedHostPlatform,
    #[error("Unsupported Android target triple: {target_triple}")]
    UnsupportedAndroidTarget { target_triple: String },
}

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
    /// # Errors
    ///
    /// Returns [`AndroidNdkDetectionError`] when the NDK installation cannot be
    /// located, an environment variable is missing, the target triple is
    /// unsupported, or the host platform is not supported by the NDK.
    pub fn detect(target_triple: &str) -> Result<Self, AndroidNdkDetectionError> {
        let ndk_path = detect_ndk_path(target_triple)?;

        validate_ndk_installation(&ndk_path)?;

        let api_level = detect_api_level();
        let abi = target_triple_to_abi(target_triple)?;
        let host_tag = detect_host_tag()?;
        let target_prefix = target_triple_to_ndk_prefix(target_triple)?;
        let toolchain_path = format!("{ndk_path}/toolchains/llvm/prebuilt/{host_tag}");

        if !Path::new(&toolchain_path).exists() {
            return Err(AndroidNdkDetectionError::NdkToolchainDirectoryMissing {
                path: PathBuf::from(toolchain_path),
            });
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

fn detect_ndk_path(target_triple: &str) -> Result<String, AndroidNdkDetectionError> {
    env::var("ANDROID_NDK")
        .or_else(|_android_ndk_unset| env::var("ANDROID_NDK_ROOT"))
        .or_else(|_android_ndk_root_unset| env::var("NDK_ROOT"))
        .or_else(|_ndk_root_unset| env::var("CARGO_NDK_ANDROID_NDK"))
        .or_else(|_cargo_ndk_android_ndk_unset| detect_ndk_from_sdk())
        .map_err(|source| AndroidNdkDetectionError::NdkRootNotConfigured {
            target_triple: target_triple.to_owned(),
            source,
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

fn validate_ndk_installation(ndk_path: &str) -> Result<(), AndroidNdkDetectionError> {
    let ndk_path = Path::new(ndk_path);

    if !ndk_path.exists() {
        return Err(AndroidNdkDetectionError::NdkRootMissing {
            path: ndk_path.to_path_buf(),
        });
    }

    let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");

    if !toolchain_file.exists() {
        return Err(AndroidNdkDetectionError::NdkToolchainFileMissing {
            path: toolchain_file,
        });
    }

    Ok(())
}

fn detect_api_level() -> String {
    env::var("ANDROID_API_LEVEL")
        .or_else(|_android_api_level_unset| {
            env::var("ANDROID_PLATFORM").map(|platform| platform.replace("android-", ""))
        })
        .or_else(|_android_platform_unset| {
            env::var("CARGO_NDK_ANDROID_PLATFORM").map(|platform| platform.replace("android-", ""))
        })
        .unwrap_or_else(|_no_api_level_configured| DEFAULT_ANDROID_API_LEVEL.to_string())
}

fn detect_host_tag() -> Result<&'static str, AndroidNdkDetectionError> {
    if cfg!(target_os = "macos") {
        Ok("darwin-x86_64")
    } else if cfg!(target_os = "linux") {
        Ok("linux-x86_64")
    } else if cfg!(target_os = "windows") {
        Ok("windows-x86_64")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedHostPlatform)
    }
}

fn target_triple_to_abi(target_triple: &str) -> Result<&'static str, AndroidNdkDetectionError> {
    if target_triple.contains("aarch64") {
        Ok("arm64-v8a")
    } else if target_triple.contains("armv7") {
        Ok("armeabi-v7a")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64")
    } else if target_triple.contains("i686") {
        Ok("x86")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedAndroidTarget {
            target_triple: target_triple.to_owned(),
        })
    }
}

fn target_triple_to_ndk_prefix(
    target_triple: &str,
) -> Result<&'static str, AndroidNdkDetectionError> {
    if target_triple.contains("aarch64") {
        Ok("aarch64-linux-android")
    } else if target_triple.contains("armv7") {
        Ok("arm-linux-androideabi")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64-linux-android")
    } else if target_triple.contains("i686") {
        Ok("i686-linux-android")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedAndroidTarget {
            target_triple: target_triple.to_owned(),
        })
    }
}

fn find_clang_builtin_includes(toolchain_path: &str) -> Option<String> {
    let clang_lib_path = format!("{toolchain_path}/lib/clang");
    let entries = std::fs::read_dir(&clang_lib_path).ok()?;

    let version_dir = entries.filter_map(std::result::Result::ok).find(|entry| {
        entry
            .file_type()
            .map(|file_type| file_type.is_dir())
            .unwrap_or(false)
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(|character: char| character.is_ascii_digit()))
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
