use std::env;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    let mut ggml_cuda = if cublas_enabled {
        Some(cc::Build::new())
    } else {
        None
    };

    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    let mut ggml = cc::Build::new();
    let mut llama_cpp = cc::Build::new();

    ggml.cpp(false);
    llama_cpp.cpp(true);

    // https://github.com/ggerganov/llama.cpp/blob/a836c8f534ab789b02da149fbdaf7735500bff74/Makefile#L364-L368
    if let Some(ggml_cuda) = &mut ggml_cuda {
        for lib in [
            "cuda", "cublas", "culibos", "cudart", "cublasLt", "pthread", "dl", "rt",
        ] {
            println!("cargo:rustc-link-lib={}", lib);
        }
        if !ggml_cuda.get_compiler().is_like_msvc() {
            for lib in ["culibos", "pthread", "dl", "rt"] {
                println!("cargo:rustc-link-lib={}", lib);
            }
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

        if cfg!(target_arch = "aarch64") {
            ggml_cuda
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            ggml.flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            llama_cpp
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            ggml.flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
        }

        ggml_cuda
            .cuda(true)
            .flag("-arch=all")
            .file("llama.cpp/ggml-cuda.cu")
            .include("llama.cpp");

        if ggml_cuda.get_compiler().is_like_msvc() {
            ggml_cuda.std("c++14");
        } else {
            ggml_cuda.flag("-std=c++11").std("c++11");
        }

        ggml.define("GGML_USE_CUBLAS", None);
        ggml_cuda.define("GGML_USE_CUBLAS", None);
        llama_cpp.define("GGML_USE_CUBLAS", None);
    }

    for build in [&mut ggml, &mut llama_cpp] {
        let compiler = build.get_compiler();

        if cfg!(target_arch = "i686") || cfg!(target_arch = "x86_64") {
            let features = x86::Features::get_target();
            if compiler.is_like_clang() || compiler.is_like_gnu() {
                build.flag("-pthread");

                if features.avx {
                    build.flag("-mavx");
                }
                if features.avx2 {
                    build.flag("-mavx2");
                }
                if features.fma {
                    build.flag("-mfma");
                }
                if features.f16c {
                    build.flag("-mf16c");
                }
                if features.sse3 {
                    build.flag("-msse3");
                }
            } else if compiler.is_like_msvc() {
                match (features.avx2, features.avx) {
                    (true, _) => {
                        build.flag("/arch:AVX2");
                    }
                    (_, true) => {
                        build.flag("/arch:AVX");
                    }
                    _ => {}
                }
            }
        } else if cfg!(target_arch = "aarch64")
            && (compiler.is_like_clang() || compiler.is_like_gnu())
        {
            if cfg!(target_os = "macos") {
                build.flag("-mcpu=apple-m1");
            } else if env::var("HOST") == env::var("TARGET") {
                build.flag("-mcpu=native");
            }
            build.flag("-pthread");
        }
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
        println!("cargo:rustc-link-lib=framework=Accelerate");

        metal_hack(&mut ggml);
        ggml.include("./llama.cpp/ggml-metal.h");
    }

    if cfg!(target_os = "dragonfly") {
        llama_cpp.define("__BSD_VISIBLE", None);
    }

    if cfg!(target_os = "linux") {
        ggml.define("_GNU_SOURCE", None);
    }

    ggml.std("c11")
        .include("./llama.cpp")
        .file("llama.cpp/ggml.c")
        .file("llama.cpp/ggml-alloc.c")
        .file("llama.cpp/ggml-backend.c")
        .file("llama.cpp/ggml-quants.c")
        .define("GGML_USE_K_QUANTS", None);

    llama_cpp
        .define("_XOPEN_SOURCE", Some("600"))
        .include("llama.cpp")
        .std("c++11")
        .file("llama.cpp/llama.cpp")
        .file("llama.cpp/unicode.cpp");

    // Remove debug log output from `llama.cpp`
    let is_release = env::var("PROFILE").unwrap() == "release";
    if is_release {
        ggml.define("NDEBUG", None);
        llama_cpp.define("NDEBUG", None);
        if let Some(cuda) = ggml_cuda.as_mut() {
            cuda.define("NDEBUG", None);
        }
    }

    if let Some(ggml_cuda) = ggml_cuda {
        println!("compiling ggml-cuda");
        ggml_cuda.compile("ggml-cuda");
        println!("compiled ggml-cuda");
    }

    println!("compiling ggml");
    ggml.compile("ggml");
    println!("compiled ggml");

    println!("compiling llama");
    llama_cpp.compile("llama");
    println!("compiled llama");

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
fn metal_hack(build: &mut cc::Build) {
    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";
    const GGML_COMMON_PATH: &str = "llama.cpp/ggml-common.h";

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not defined"));

    let ggml_metal_path = {
        let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
            .expect("Could not read ggml-metal.metal")
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\"', "\\\"");

        let ggml_common = std::fs::read_to_string(GGML_COMMON_PATH).expect("Could not read ggml-common.h")
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\"', "\\\"");

        let includged_ggml_metal_metal = ggml_metal_metal.replace(
            "#include \\\"ggml-common.h\\\"",
            &format!("{ggml_common}")
        );
        print!("{}", &includged_ggml_metal_metal);

        let ggml_metal =
            std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

        let needle = r#"NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];"#;
        if !ggml_metal.contains(needle) {
            panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated. Contact a `llama-cpp-sys-2` developer!");
        }

        // Replace the runtime read of the file with a compile-time string
        let ggml_metal = ggml_metal.replace(
            needle,
            &format!(r#"NSString * src  = @"{includged_ggml_metal_metal}";"#),
        );

        let patched_ggml_metal_path = out_dir.join("ggml-metal.m");
        std::fs::write(&patched_ggml_metal_path, ggml_metal)
            .expect("Could not write temporary patched ggml-metal.m");

        patched_ggml_metal_path
    };

    build.file(ggml_metal_path);
}

// Courtesy of https://github.com/rustformers/llm
fn get_supported_target_features() -> std::collections::HashSet<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(ToString::to_string)
        .collect()
}

mod x86 {
    #[allow(clippy::struct_excessive_bools)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Features {
        pub fma: bool,
        pub avx: bool,
        pub avx2: bool,
        pub f16c: bool,
        pub sse3: bool,
    }
    impl Features {
        pub fn get_target() -> Self {
            let features = crate::get_supported_target_features();
            Self {
                fma: features.contains("fma"),
                avx: features.contains("avx"),
                avx2: features.contains("avx2"),
                f16c: features.contains("f16c"),
                sse3: features.contains("sse3"),
            }
        }
    }
}
