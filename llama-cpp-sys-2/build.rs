use std::env;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    let mut ggml = cc::Build::new();
    let mut llama_cpp = cc::Build::new();

    ggml.cpp(false);
    llama_cpp.cpp(true);

    // https://github.com/ggerganov/llama.cpp/blob/a836c8f534ab789b02da149fbdaf7735500bff74/Makefile#L364-L368
    if cublas_enabled {
        for lib in [
            "cuda", "cublas", "culibos", "cudart", "cublasLt", "pthread", "dl", "rt",
        ] {
            println!("cargo:rustc-link-lib={}", lib);
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

        if cfg!(target_arch = "aarch64") {
            ggml.flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            llama_cpp
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            ggml.flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
        }

        ggml.cuda(true)
            .std("c++17")
            .flag("-arch=all")
            .file("llama.cpp/ggml-cuda.cu");

        ggml.define("GGML_USE_CUBLAS", None);
        ggml.define("GGML_USE_CUBLAS", None);
        llama_cpp.define("GGML_USE_CUBLAS", None);
    }

    // https://github.com/ggerganov/llama.cpp/blob/191221178f51b6e81122c5bda0fd79620e547d07/Makefile#L133-L141
    if cfg!(target_os = "macos") {
        assert!(!cublas_enabled, "CUBLAS is not supported on macOS");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalKit");

        llama_cpp.define("_DARWIN_C_SOURCE", None);

        // https://github.com/ggerganov/llama.cpp/blob/3c0d25c4756742ebf15ad44700fabc0700c638bd/Makefile#L340-L343
        llama_cpp.define("GGML_USE_METAL", None);
        llama_cpp.define("GGML_USE_ACCELERATE", None);
        llama_cpp.define("ACCELERATE_NEW_LAPACK", None);
        llama_cpp.define("ACCELERATE_LAPACK_ILP64", None);
        println!("cargo:rustc-link-arg=framework=Accelerate");

        // 	MK_LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit
        // https://github.com/ggerganov/llama.cpp/blob/3c0d25c4756742ebf15ad44700fabc0700c638bd/Makefile#L509-L511
        println!("cargo:rustc-link-arg=framework=Foundation");
        println!("cargo:rustc-link-arg=framework=Metal");
        println!("cargo:rustc-link-arg=framework=MetalKit");

        metal_hack(&mut ggml);
        ggml.include("./llama.cpp/ggml-metal.h");
    }

    if cfg!(target_os = "dragonfly") {
        llama_cpp.define("__BSD_VISIBLE", None);
    }

    if cfg!(target_os = "linux") {
        ggml.define("_GNU_SOURCE", None);
    }

    ggml.std("c17")
        .include("./llama.cpp")
        .file("llama.cpp/ggml.c")
        .file("llama.cpp/ggml-alloc.c")
        .file("llama.cpp/ggml-backend.c")
        .file("llama.cpp/ggml-quants.c")
        .define("GGML_USE_K_QUANTS", None);

    llama_cpp
        .define("_XOPEN_SOURCE", Some("600"))
        .include("llama.cpp")
        .std("c++17")
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

// courtesy of https://github.com/rustformers/llm
fn metal_hack(build: &mut cc::Build) {
    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not defined"));

    let ggml_metal_path = {
        let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
            .expect("Could not read ggml-metal.metal")
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\"', "\\\"");

        let ggml_metal =
            std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

        let needle = r#"NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error];"#;
        if !ggml_metal.contains(needle) {
            panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated. Contact a `llama-cpp-sys-2` developer!");
        }

        // Replace the runtime read of the file with a compile-time string
        let ggml_metal = ggml_metal.replace(
            needle,
            &format!(r#"NSString * src  = @"{ggml_metal_metal}";"#),
        );

        let patched_ggml_metal_path = out_dir.join("ggml-metal.m");
        std::fs::write(&patched_ggml_metal_path, ggml_metal)
            .expect("Could not write temporary patched ggml-metal.m");

        patched_ggml_metal_path
    };

    build.file(ggml_metal_path);
}
