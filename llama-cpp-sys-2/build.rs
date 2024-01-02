use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    let mut cmake_build = cmake::Config::new("llama.cpp");

    if cublas_enabled {
        cmake_build.define("LLAMA_CUBLAS", "ON");
    }

    cmake_build.define("LLAMA_STATIC", "ON");
    cmake_build.build_target("llama");

    let llama = cmake_build.build();

    println!("cargo:rustc-link-lib=dylib=stdc++");

    if cublas_enabled {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
    }

    println!("cargo:rustc-link-search=native={}/build", llama.display());
    println!("cargo:rustc-link-lib=static=llama");

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
