use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    cc::Build::new()
        .include("llama.cpp")
        .file("llama.cpp/ggml.c")
        .define("_GNU_SOURCE", Some("1"))
        .compile("ggml");

    cc::Build::new()
        .cuda(true)
        .include("llama.cpp")
        .file("llama.cpp/ggml-cuda.cu")
        .compile("ggml-cuda");

    cc::Build::new()
        .include("llama.cpp")
        .file("llama.cpp/ggml-alloc.c")
        .compile("ggml-alloc");

    cc::Build::new()
        .include("llama.cpp")
        .file("llama.cpp/ggml-backend.c")
        .compile("ggml-backend");

    cc::Build::new()
        .include("llama.cpp")
        .file("llama.cpp/ggml-quants.c")
        .compile("ggml-quants");

    cc::Build::new()
        .cpp(true)
        .include("llama.cpp")
        .define("GGML_USE_CUBLAS", if cublas_enabled { Some("1") } else { None })
        .cuda(cublas_enabled)
        .file("llama.cpp/llama.cpp")
        .compile("llama");

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
