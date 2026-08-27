//! Pre-generate bindings to `llama.cpp`.
//!
//! This is done to reduce compile-times, as it allows downstream users to
//! avoid having to depend on `bindgen`.

use std::path::Path;

use bindgen::{
    RustTarget,
    callbacks::{DeriveTrait, ImplementsTrait, ParseCallbacks},
};

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent().unwrap();
    let sys = workspace.join("llama-cpp-sys-2");
    let llama = sys.join("llama.cpp");

    #[allow(deprecated)]
    let rust_target = RustTarget::Stable_1_77;

    let mut builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", llama.join("include").display()))
        .clang_arg(format!("-I{}", llama.join("ggml/include").display()))
        .clang_arg(format!("-I{}", llama.join("tools/mtmd/include").display()))
        .derive_partialeq(true)
        .prepend_enum_name(false)
        .rust_target(rust_target)
        // These differ between 32- and 64-bit targets, so we can't
        // pre-generate them.
        .layout_tests(false)
        // We'd like to split things into different modules.
        .allowlist_recursively(false)
        .raw_line("use crate::*;")
        .parse_callbacks(Box::new(AllBlocklistedImplementsTrait));

    // Fix bindgen header discovery on Windows MSVC.
    if cfg!(target_env = "msvc") {
        let host_arch = env!("HOST_TARGET_ARCH");
        let tool =
            find_msvc_tools::find_tool(host_arch, "cl.exe").expect("could not find MSVC tool");

        // Extract include paths by checking compiler's environment
        // cc crate sets up MSVC environment internally
        let env_include = tool
            .env()
            .into_iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("INCLUDE"))
            .map(|(_, v)| v);

        if let Some(include_paths) = env_include {
            for include_path in include_paths
                .to_string_lossy()
                .split(';')
                .filter(|s| !s.is_empty())
            {
                builder = builder.clang_arg("-isystem").clang_arg(include_path);
            }
        }

        // Add MSVC compatibility flags
        builder = builder.clang_arg("-fms-compatibility");
        builder = builder.clang_arg("-fms-extensions");
    }

    // builder = builder.clang_arg("-target").clang_arg("i686-apple-macosx");

    let headers = [
        // FIXME(madsmtm): Split `ggml.rs` up more?
        (llama.join("include/llama.h"), "ggml.rs", "ggml_.*"),
        (llama.join("ggml/include/gguf.h"), "gguf.rs", "gguf_.*"),
        (llama.join("include/llama.h"), "llama.rs", "llama_.*"),
        (llama.join("tools/mtmd/mtmd-helper.h"), "mtmd.rs", "mtmd_.*"),
        // The `llama_rs_*` symbols are emitted by `wrapper_common.cpp`.
        (sys.join("wrapper_common.h"), "common.rs", "llama_rs_.*"),
    ];

    for (header, output, allowlist_item) in headers {
        builder
            .clone()
            .allowlist_item(allowlist_item)
            .header(header.to_str().unwrap())
            .generate()
            .expect("failed to generate bindings")
            .write_to_file(sys.join("src").join(output))
            .expect("failed writing binding file");
    }
}

/// Workaround: We split bindings into files, but that means that each file
/// has to blocklist the items in the other files. So we have to tell bindgen
/// that they actually implement all the traits we want.
#[derive(Debug)]
struct AllBlocklistedImplementsTrait;

impl ParseCallbacks for AllBlocklistedImplementsTrait {
    fn blocklisted_type_implements_trait(
        &self,
        _name: &str,
        _derive_trait: DeriveTrait,
    ) -> Option<ImplementsTrait> {
        Some(ImplementsTrait::Yes)
    }
}
