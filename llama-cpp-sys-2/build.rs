use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let build = Config::new("llama.cpp")
        .define("LLAMA_CUBLAS", if cfg!(feature = "cublas") { "ON" } else { "OFF" })
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
}