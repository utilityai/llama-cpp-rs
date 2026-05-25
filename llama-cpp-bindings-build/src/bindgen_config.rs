use std::env;
use std::path::Path;

use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::target_os::TargetOs;

pub fn generate_bindings(
    llama_src: &Path,
    out_dir: &Path,
    target_os: &TargetOs,
    target_triple: &str,
    android_ndk: Option<&AndroidNdk>,
) {
    let mut builder = create_base_builder(llama_src);

    if target_os.is_android()
        && let Some(ndk) = android_ndk
    {
        builder = configure_android_bindgen(builder, ndk, target_triple);
    }

    if target_os.is_msvc() {
        builder = configure_msvc_bindgen(builder, target_triple);
    }

    let bindings = builder
        .generate()
        .expect("bindgen failed to generate FFI bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write generated bindings to file");

    debug_log!("Bindings Created");
}

fn create_base_builder(llama_src: &Path) -> bindgen::Builder {
    bindgen::Builder::default()
        .header("wrapper.h")
        .header("wrapper_mtmd.h")
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("llama_rs_.*")
        .allowlist_type("llama_rs_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .blocklist_function("ggml_fopen")
        .blocklist_function("gguf_init_from_file_ptr")
        .blocklist_function("gguf_write_to_file_ptr")
        .blocklist_function("llama_model_load_from_file_ptr")
        .blocklist_type("FILE")
        .blocklist_type("_IO_.*")
        .blocklist_type("_iobuf")
        .blocklist_type("__BindgenBitfieldUnit")
        .prepend_enum_name(false)
}

fn configure_android_bindgen(
    mut builder: bindgen::Builder,
    ndk: &AndroidNdk,
    target_triple: &str,
) -> bindgen::Builder {
    builder = builder
        .clang_arg(format!("--sysroot={}", ndk.sysroot))
        .clang_arg(format!("-D__ANDROID_API__={}", ndk.api_level))
        .clang_arg("-D__ANDROID__");

    if let Some(ref builtin_includes) = ndk.clang_builtin_includes {
        builder = builder.clang_arg("-isystem").clang_arg(builtin_includes);
    }

    builder = builder
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include/{}", ndk.sysroot, ndk.target_prefix))
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include", ndk.sysroot))
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        .clang_arg("-include")
        .clang_arg("stdint.h");

    if env::var("CARGO_SUBCOMMAND").as_deref() == Ok("ndk") {
        // SAFETY: build scripts are single-threaded, so modifying env is safe.
        unsafe {
            env::set_var(
                "BINDGEN_EXTRA_CLANG_ARGS",
                format!("--target={target_triple}"),
            );
        }
    }

    builder
}

fn configure_msvc_bindgen(mut builder: bindgen::Builder, target_triple: &str) -> bindgen::Builder {
    let out_dir_str = env::var("OUT_DIR").unwrap_or_default();
    let dummy_c = Path::new(&out_dir_str).join("dummy.c");

    if std::fs::write(&dummy_c, "int main() { return 0; }").is_err() {
        return builder;
    }

    let mut cc_build = cc::Build::new();
    cc_build.file(&dummy_c);

    let Ok(compiler) = cc_build.try_get_compiler() else {
        return builder;
    };

    let msvc_include_paths = compiler
        .env()
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("INCLUDE"))
        .map(|(_, value)| value.clone());

    if let Some(include_paths) = msvc_include_paths {
        for include_path in include_paths
            .to_string_lossy()
            .split(';')
            .filter(|path| !path.is_empty())
        {
            builder = builder.clang_arg("-isystem").clang_arg(include_path);
            debug_log!("Added MSVC include path: {}", include_path);
        }
    }

    builder = builder
        .clang_arg(format!("--target={target_triple}"))
        .clang_arg("-fms-compatibility")
        .clang_arg("-fms-extensions");

    debug_log!(
        "Configured bindgen with MSVC toolchain for target: {}",
        target_triple
    );

    builder
}
