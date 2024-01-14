use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    let mut ggml = cc::Build::new();
    let mut ggml_cuda = if cublas_enabled { Some(cc::Build::new()) } else { None };
    let mut llama_cpp = cc::Build::new();

    ggml.cpp(false);
    llama_cpp.cpp(true);

    if let Some(ggml_cuda) = &mut ggml_cuda {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");

        let libs = "cuda cublas culibos cudart cublasLt pthread dl rt";

        for lib in libs.split_whitespace() {
            println!("cargo:rustc-link-lib={}", lib);
        }

        ggml_cuda
            .cuda(true)
            .flag("-arch=native")
            .file("llama.cpp/ggml-cuda.cu")
            .include("llama.cpp/ggml-cuda.h");

        ggml.define("GGML_USE_CUBLAS", None);
        ggml_cuda.define("GGML_USE_CUBLAS", None);
        llama_cpp.define("GGML_USE_CUBLAS", None);
    }

    if let Some(ggml_cuda) = ggml_cuda {
        println!("compiling ggml-cuda");
        ggml_cuda.compile("ggml-cuda");
    }

    ggml
        .file("llama.cpp/ggml.c")
        .file("llama.cpp/ggml-alloc.c")
        .file("llama.cpp/ggml-backend.c")
        .file("llama.cpp/ggml-quants.c")
        .define("_GNU_SOURCE", None)
        .define("GGML_USE_K_QUANTS", None);

    llama_cpp
        .file("llama.cpp/llama.cpp");

    println!("compiling ggml");
    ggml.compile("ggml");

    println!("compiling llama");
    llama_cpp.compile("llama");

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
