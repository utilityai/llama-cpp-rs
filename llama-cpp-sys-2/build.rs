use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    if cfg!(target_os = "macos") {
        metal_hack();
    }

    let build = Config::new("llama.cpp")
        .define(
            "LLAMA_CUBLAS",
            if cfg!(feature = "cublas") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("BUILD_SHARED_LIBS", "ON")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .build();

    let shared = build.join("lib");
    println!("cargo:rustc-link-search={}", shared.display());
    println!("cargo:rustc-link-lib=dylib=llama");

    let header = "llama.cpp/llama.h";

    println!("cargo:rerun-if-changed={header}");

    let bindings = bindgen::builder()
        .header(header)
        .derive_partialeq(true)
        .no_debug("llama_grammar_element")
        .prepend_enum_name(false)
        .derive_eq(true)
        .generate()
        .expect("failed to generate bindings for llama.cpp");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings to file");
    let llama_cpp_dir = PathBuf::from("llama.cpp").canonicalize().unwrap();
    println!("cargo:INCLUDE={}", llama_cpp_dir.to_str().unwrap());
    println!("cargo:OUT_DIR={}", out_path.to_str().unwrap());
}

// courtesy of https://github.com/rustformers/llm
fn metal_hack() {
    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";
    const GGML_COMMON_PATH: &str = "llama.cpp/ggml-common.h";

    let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
        .expect("Could not read ggml-metal.metal")
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\"', "\\\"");

    let ggml_common = std::fs::read_to_string(GGML_COMMON_PATH)
        .expect("Could not read ggml-common.h")
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\"', "\\\"");

    let includged_ggml_metal_metal =
        ggml_metal_metal.replace("#include \\\"ggml-common.h\\\"", &format!("{ggml_common}"));

    let ggml_metal = std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

    let needle = r#"NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];"#;
    if !ggml_metal.contains(needle) {
        panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated. Contact a `llama-cpp-sys-2` developer!");
    }

    // Replace the runtime read of the file with a compile-time string
    let ggml_metal = ggml_metal.replace(
        needle,
        &format!(r#"NSString * src  = @"{includged_ggml_metal_metal}";"#),
    );

    std::fs::write(&GGML_METAL_PATH, ggml_metal)
        .expect("Could not write temporary patched ggml-metal.m");
}
