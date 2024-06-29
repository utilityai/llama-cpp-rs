use std::env::{self, VarError};
use std::fs::{read_dir, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use cc::Build;
use once_cell::sync::Lazy;
use glob::glob;

// This build file is based on:
// https://github.com/mdrokz/rust-llama.cpp/blob/master/build.rs
// License MIT
// 12-2-2024

#[cfg(all(
    feature = "metal",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"metal\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "cuda",
    any(
        feature = "metal",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"cuda\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "blas",
    any(
        feature = "cuda",
        feature = "metal",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"blas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "hipblas",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "metal",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"hipblas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "clblast",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "metal",
        feature = "vulkan"
    )
))]
compile_error!("feature \"clblas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "vulkan",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "metal"
    )
))]
compile_error!("feature \"vulkan\" cannot be enabled alongside other GPU based features");

static LLAMA_PATH: Lazy<PathBuf> = Lazy::new(|| PathBuf::from("./llama.cpp"));

fn compile_bindings(out_path: &Path) {
    println!("Generating bindings..");
    let bindings = bindgen::Builder::default()
        .header(LLAMA_PATH.join("ggml.h").to_string_lossy())
        .header(LLAMA_PATH.join("llama.h").to_string_lossy())
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .prepend_enum_name(false);

    #[cfg(all(
        feature = "compat",
        not(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))
    ))]
    {
        bindings = bindings.parse_callbacks(Box::new(GGMLLinkRename {}));
    }

    let bindings = bindings.generate().expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(all(
    feature = "compat",
    not(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))
))]
#[derive(Debug)]
struct GGMLLinkRename {}

#[cfg(all(
    feature = "compat",
    not(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))
))]
impl ParseCallbacks for GGMLLinkRename {
    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        match item_info.kind {
            ItemKind::Function => {
                if item_info.name.starts_with("ggml_") {
                    Some(format!("{PREFIX}{}", item_info.name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Add platform appropriate flags and definitions present in all compilation configurations.
fn push_common_flags(cx: &mut Build, cxx: &mut Build) {
    cx.static_flag(true)
        .cpp(false)
        .define("GGML_SCHED_MAX_COPIES", "4");
    cxx.static_flag(true)
        .cpp(true)
        .define("GGML_SCHED_MAX_COPIES", "4");

    if !cfg!(debug_assertions) {
        cx.define("NDEBUG", None);
        cxx.define("NDEBUG", None);
    } else {
        cx.define("GGML_DEBUG", "100");
        cxx.define("GGML_DEBUG", "100");

        if cfg!(target_os = "linux") {
            cx.define("_GLIBCXX_ASSERTIONS", None);
            cxx.define("_GLIBCXX_ASSERTIONS", None);
        } else if cfg!(target_os = "windows") {
            cx.define("_CRT_SECURE_NO_WARNINGS", None);
            cxx.define("_CRT_SECURE_NO_WARNINGS", None);
        }
    }

    if cfg!(target_os = "openbsd") {
        cx.define("_XOPEN_SOURCE", "700");
        cxx.define("_XOPEN_SOURCE", "700");
    } else {
        cx.define("_XOPEN_SOURCE", "600");
        cxx.define("_XOPEN_SOURCE", "600");
    }

    if cfg!(target_os = "linux") {
        cx.define("_GNU_SOURCE", None);
        cxx.define("_GNU_SOURCE", None);
    } else if cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "dragonfly"
    )) {
        cx.define("_DARWIN_C_SOURCE", None);
        cxx.define("_DARWIN_C_SOURCE", None);
    } else if cfg!(target_os = "openbsd") {
        cx.define("_BSD_SOURCE", None);
        cxx.define("_BSD_SOURCE", None);
    } else if cfg!(target_os = "freebsd") {
        cx.define("__BSD_VISIBLE", None);
        cxx.define("__BSD_VISIBLE", None);
    } else if cfg!(target_os = "netbsd") {
        cx.define("_NETBSD_SOURCE", None);
        cxx.define("_NETBSD_SOURCE", None);
    }

    if cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
        if cfg!(target_family = "unix") {
            // cx.flag("-mavx512vnni").flag("-mfp16-format=ieee");
            // cxx.flag("-mavx512vnni").flag("-mfp16-format=ieee");
        } else if cfg!(target_family = "windows") {
            cx.define("__ARM_NEON", None)
                .define("__ARM_FEATURE_FMA", None)
                .define("__ARM_FEATURE_DOTPROD", None)
                .define("__aarch64__", None);
            cxx.define("__ARM_NEON", None)
                .define("__ARM_FEATURE_FMA", None)
                .define("__ARM_FEATURE_DOTPROD", None)
                .define("__aarch64__", None);
        }
    }
}

/// Add platform appropriate flags and definitions for compilation warnings.
fn push_warn_flags(cx: &mut Build, cxx: &mut Build) {
    if cfg!(target_family = "unix") {
        cx.flag("-pthread")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .flag("-Wcast-qual")
            .flag("-Wdouble-promotion")
            .flag("-Wshadow")
            .flag("-Wstrict-prototypes")
            .flag("-Wpointer-arith");
        cxx.flag("-fPIC")
            .flag("-pthread")
            .flag("-Wall")
            .flag("-Wdeprecated-declarations")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .flag("-Wcast-qual")
            .flag("-Wno-unused-function")
            .flag("-Wno-multichar");
    } else if cfg!(target_family = "windows") {
        cx.flag("/W4")
            .flag("/Wall")
            .flag("/wd4820")
            .flag("/wd4710")
            .flag("/wd4711")
            .flag("/wd4820")
            .flag("/wd4514");
        cxx.flag("/W4")
            .flag("/Wall")
            .flag("/wd4820")
            .flag("/wd4710")
            .flag("/wd4711")
            .flag("/wd4820")
            .flag("/wd4514");
    }
}

/// Add platform appropriate flags and definitions based on enabled features.
fn push_feature_flags(cx: &mut Build, cxx: &mut Build) {
    // TODO in llama.cpp's cmake (https://github.com/ggerganov/llama.cpp/blob/9ecdd12e95aee20d6dfaf5f5a0f0ce5ac1fb2747/CMakeLists.txt#L659), they include SIMD instructions manually, however it doesn't seem to be necessary for VS2022's MSVC, check when it is needed

    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        if cfg!(feature = "native") && cfg!(target_os = "linux") {
            cx.flag("-march=native");
            cxx.flag("-march=native");
        }

        if cfg!(feature = "fma") && cfg!(target_family = "unix") {
            cx.flag("-mfma");
            cxx.flag("-mfma");
        }

        if cfg!(feature = "f16c") && cfg!(target_family = "unix") {
            cx.flag("-mf16c");
            cxx.flag("-mf16c");
        }

        if cfg!(target_family = "unix") {
            if cfg!(feature = "avx512") {
                cx.flag("-mavx512f").flag("-mavx512bw");
                cxx.flag("-mavx512f").flag("-mavx512bw");

                if cfg!(feature = "avx512_vmbi") {
                    cx.flag("-mavx512vbmi");
                    cxx.flag("-mavx512vbmi");
                }

                if cfg!(feature = "avx512_vnni") {
                    cx.flag("-mavx512vnni");
                    cxx.flag("-mavx512vnni");
                }
            }

            if cfg!(feature = "avx2") {
                cx.flag("-mavx2");
                cxx.flag("-mavx2");
            }

            if cfg!(feature = "avx") {
                cx.flag("-mavx");
                cxx.flag("-mavx");
            }
        } else if cfg!(target_family = "windows") {
            if cfg!(feature = "avx512") {
                cx.flag("/arch:AVX512");
                cxx.flag("/arch:AVX512");

                if cfg!(feature = "avx512_vmbi") {
                    cx.define("__AVX512VBMI__", None);
                    cxx.define("__AVX512VBMI__", None);
                }

                if cfg!(feature = "avx512_vnni") {
                    cx.define("__AVX512VNNI__", None);
                    cxx.define("__AVX512VNNI__", None);
                }
            } else if cfg!(feature = "avx2") {
                cx.flag("/arch:AVX2");
                cxx.flag("/arch:AVX2");
            } else if cfg!(feature = "avx") {
                cx.flag("/arch:AVX");
                cxx.flag("/arch:AVX");
            }
        }
    }
}

fn compile_opencl(cx: &mut Build, cxx: &mut Build) {
    println!("Compiling OpenCL GGML..");

    // TODO
    println!("cargo:warning=OpenCL compilation and execution has not been properly tested yet");

    cx.define("GGML_USE_CLBLAST", None);
    cxx.define("GGML_USE_CLBLAST", None);

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    }

    cxx.file(LLAMA_PATH.join("ggml-opencl.cpp"));
}

fn compile_openblas(cx: &mut Build) {
    println!("Compiling OpenBLAS GGML..");

    // TODO
    println!("cargo:warning=OpenBlas compilation and execution has not been properly tested yet");

    cx.define("GGML_USE_OPENBLAS", None)
        .include("/usr/local/include/openblas")
        .include("/usr/local/include/openblas");
    println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(cx: &mut Build) {
    println!("Compiling BLIS GGML..");

    // TODO
    println!("cargo:warning=Blis compilation and execution has not been properly tested yet");

    cx.define("GGML_USE_OPENBLAS", None)
        .include("/usr/local/include/blis")
        .include("/usr/local/include/blis");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=blis");
}

fn compile_hipblas(cx: &mut Build, cxx: &mut Build, mut hip: Build) -> &'static str {
    let rocm_path_str = env::var("ROCM_PATH").or(Ok::<String, VarError>(String::from_str("/opt/rocm/").unwrap())).unwrap();

    println!("Compiling hipBLAS GGML. Using ROCm from {rocm_path_str}");

    let rocm_path = PathBuf::from(rocm_path_str);
    let rocm_include = rocm_path.join("include");
    let rocm_lib = rocm_path.join("lib");
    let rocm_hip_bin = rocm_path.join("bin/hipcc");

    let defines = ["GGML_USE_HIPBLAS", "GGML_USE_CUDA"];
    for def in defines {
        cx.define(def, None);
        cxx.define(def, None);
    }

    cx.include(&rocm_include);
    cxx.include(&rocm_include);

    let ggml_cuda = glob(LLAMA_PATH.join("ggml-cuda").join("*.cu").to_str().unwrap())
        .unwrap().filter_map(Result::ok).collect::<Vec<_>>();
    let ggml_template_fattn = glob(LLAMA_PATH.join("ggml-cuda").join("template-instances").join("fattn-vec*.cu").to_str().unwrap())
        .unwrap().filter_map(Result::ok).collect::<Vec<_>>();
    let ggml_template_wmma = glob(LLAMA_PATH.join("ggml-cuda").join("template-instances").join("fattn-wmma*.cu").to_str().unwrap())
        .unwrap().filter_map(Result::ok).collect::<Vec<_>>();
    let ggml_template_mmq = glob(LLAMA_PATH.join("ggml-cuda").join("template-instances").join("mmq*.cu").to_str().unwrap())
        .unwrap().filter_map(Result::ok).collect::<Vec<_>>();

    hip.compiler(rocm_hip_bin)
        .std("c++11")
        .define("LLAMA_CUDA_DMMV_X", Some("32"))
        .define("LLAMA_CUDA_MMV_Y", Some("1"))
        .define("LLAMA_CUDA_KQUANTS_ITER", Some("2"))
        .file(LLAMA_PATH.join("ggml-cuda.cu"))
        .files(ggml_cuda)
        .files(ggml_template_fattn)
        .files(ggml_template_wmma)
        .files(ggml_template_mmq)
        .include(LLAMA_PATH.join(""))
        .include(LLAMA_PATH.join("ggml-cuda"))
        .define("GGML_USE_HIPBLAS", None)
        .define("GGML_USE_CUDA", None)
        .compile("ggml-cuda");

    println!("cargo:rustc-link-search=native={}", rocm_lib.to_string_lossy());

    let rocm_libs = ["hipblas", "rocblas", "amdhip64"];
    for lib in rocm_libs {
        println!("cargo:rustc-link-lib={lib}");
    }

    "ggml-cuda"
}

fn compile_cuda(cx: &mut Build, cxx: &mut Build, featless_cxx: Build) -> &'static str {
    println!("Compiling CUDA GGML..");

    cx.define("GGML_USE_CUDA", None);
    cxx.define("GGML_USE_CUDA", None);

    let mut nvcc = featless_cxx;
    nvcc.cuda(true)
        .flag("--forward-unknown-to-host-compiler")
        .flag("-arch=all")
        .define("K_QUANTS_PER_ITERATION", Some("2"))
        .define("GGML_CUDA_PEER_MAX_BATCH_SIZE", Some("128"));

    // if cfg!(target_os = "linux") {
    //     nvcc.flag("-Wno-pedantic");
    // }

    for lib in [
        "cuda", "cublas", "culibos", "cudart", "cublasLt", "pthread", "dl", "rt",
    ] {
        println!("cargo:rustc-link-lib={}", lib);
    }
    if !nvcc.get_compiler().is_like_msvc() {
        for lib in ["culibos", "pthread", "dl", "rt"] {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    if nvcc.get_compiler().is_like_msvc() {
        nvcc.std("c++14");
    } else {
        nvcc.flag("-std=c++11").std("c++11");
    }

    if cfg!(feature = "cuda_dmmv") {
        nvcc.define("GGML_CUDA_FORCE_DMMV", None)
            .define("GGML_CUDA_DMMV_X", Some("32"))
            .define("GGML_CUDA_MMV_Y", Some("1"));
    }

    if cfg!(feature = "cuda_mmq") {
        nvcc.define("GGML_CUDA_FORCE_MMQ", None);
    }

    let lib_name = "ggml-cuda";
    let cuda_path = LLAMA_PATH.join("ggml-cuda");
    let cuda_sources = read_dir(cuda_path.as_path())
        .unwrap()
        .map(|f| f.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".cu"))
        .map(|entry| entry.path());
    
    let template_instances = read_dir(cuda_path.join("template-instances"))
        .unwrap()
        .map(|f| f.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".cu"))
        .map(|entry| entry.path());

    nvcc.include(cuda_path.as_path())
        .include(LLAMA_PATH.as_path())
        .files(cuda_sources)
        .files(template_instances)
        .file(LLAMA_PATH.join("ggml-cuda.cu"))
        .compile(lib_name);

    lib_name
}

fn compile_metal(cx: &mut Build, cxx: &mut Build) {
    println!("Compiling Metal GGML..");

    cx.define("GGML_USE_METAL", None);
    cxx.define("GGML_USE_METAL", None);

    cx.define("GGML_METAL_EMBED_LIBRARY", None);
    cxx.define("GGML_METAL_EMBED_LIBRARY", None);

    if !cfg!(debug_assertions) {
        cx.define("GGML_METAL_NDEBUG", None);
    }

    // It's idiomatic to use OUT_DIR for intermediate c/c++ artifacts
    let out_dir = env::var("OUT_DIR").unwrap();

    let ggml_metal_shader_path = LLAMA_PATH.join("ggml-metal.metal");

    // Create a temporary assembly file that will allow for static linking to the metal shader.
    let ggml_metal_embed_assembly_path = PathBuf::from(&out_dir).join("ggml-metal-embed.asm");
    let mut ggml_metal_embed_assembly_file = File::create(&ggml_metal_embed_assembly_path)
        .expect("Failed to open ggml-metal-embed.asm file");

    let ggml_metal_shader_out_path = PathBuf::from(&out_dir).join("ggml-metal.metal");
    let common = LLAMA_PATH.join("ggml-common.h");

    let input_file = File::open(ggml_metal_shader_path).expect("Failed to open input file");
    let output_file = File::create(&ggml_metal_shader_out_path).expect("Failed to create output file");

    let output = Command::new("sed")
        .arg("-e")
        .arg(format!(
            "/#include \"ggml-common.h\"/r {}",
            common.to_string_lossy()
        ))
        .arg("-e")
        .arg("/#include \"ggml-common.h\"/d")
        .stdin(input_file)
        .stdout(output_file)
        .output()
        .expect("Failed to execute command");
    if !output.status.success() {
        panic!(
            "An error has occurred while embedding common file ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // The contents of this file is directly copied from the llama.cpp Makefile
    let ggml_metal_embed_assembly_code = format!(
        ".section __DATA, __ggml_metallib\n\
         .globl _ggml_metallib_start\n\
         _ggml_metallib_start:\n\
         .incbin \"{}\"\n\
         .globl _ggml_metallib_end\n\
         _ggml_metallib_end:\n",
        ggml_metal_shader_out_path
            .to_str()
            .expect("Failed to convert path to string")
    );

    write!(
        ggml_metal_embed_assembly_file,
        "{}",
        ggml_metal_embed_assembly_code
    )
    .expect("Failed to write ggml metal embed assembly code");

    // Assemble the ggml metal embed code.
    let ggml_metal_embed_object_path = PathBuf::from(&out_dir).join("ggml-metal-embed.o");
    Command::new("as")
        .arg(&ggml_metal_embed_assembly_path)
        .arg("-o")
        .arg(&ggml_metal_embed_object_path)
        .status()
        .expect("Failed to assemble ggml-metal-embed file");

    // Create a static library for our metal embed code.
    let ggml_metal_embed_library_path = PathBuf::from(&out_dir).join("libggml-metal-embed.a");
    Command::new("ar")
        .args([
            "crus",
            ggml_metal_embed_library_path.to_str().unwrap(),
            ggml_metal_embed_object_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to create static library from ggml-metal-embed object file");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    // Link to our new static library for our metal embed code.
    println!("cargo:rustc-link-search=native={}", &out_dir);
    println!("cargo:rustc-link-lib=static=ggml-metal-embed");

    cx.include(LLAMA_PATH.join("ggml-metal.h"))
        .file(LLAMA_PATH.join("ggml-metal.m"));
}

fn compile_vulkan(cx: &mut Build, cxx: &mut Build) -> &'static str {
    println!("Compiling Vulkan GGML..");

    // Vulkan gets linked through the ash crate.

    if cfg!(debug_assertions) {
        cx.define("GGML_VULKAN_DEBUG", None)
            .define("GGML_VULKAN_CHECK_RESULTS", None)
            .define("GGML_VULKAN_VALIDATE", None);
        cxx.define("GGML_VULKAN_DEBUG", None)
            .define("GGML_VULKAN_CHECK_RESULTS", None)
            .define("GGML_VULKAN_VALIDATE", None);
    }

    cx.define("GGML_USE_VULKAN", None);
    cxx.define("GGML_USE_VULKAN", None);

    let lib_name = "ggml-vulkan";

    cxx.clone()
        .include("./thirdparty/Vulkan-Headers/include/")
        .include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("ggml-vulkan.cpp"))
        .compile(lib_name);

    lib_name
}

fn compile_ggml(mut cx: Build) {
    println!("Compiling GGML..");
    cx.std("c11")
        .include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("ggml.c"))
        .file(LLAMA_PATH.join("ggml-alloc.c"))
        .file(LLAMA_PATH.join("ggml-backend.c"))
        .file(LLAMA_PATH.join("ggml-quants.c"))
        .compile("ggml");
}

fn compile_llama(mut cxx: Build, _out_path: impl AsRef<Path>) {
    println!("Compiling Llama.cpp..");
    cxx.std("c++11")
        .include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("unicode.cpp"))
        .file(LLAMA_PATH.join("unicode-data.cpp"))
        .file(LLAMA_PATH.join("llama.cpp"))
        .compile("llama");
}

fn main() {
    if std::fs::read_dir(LLAMA_PATH.as_path()).is_err() {
        panic!(
            "Could not find {}. Did you forget to initialize submodules?",
            LLAMA_PATH.display()
        );
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    println!("cargo:rerun-if-changed={}", LLAMA_PATH.display());

    compile_bindings(&out_path);

    let mut cx = Build::new();
    let mut cxx = Build::new();

    push_common_flags(&mut cx, &mut cxx);

    let featless_cxx = cxx.clone(); // mostly used for CUDA

    push_warn_flags(&mut cx, &mut cxx);
    push_feature_flags(&mut cx, &mut cxx);

    let _feat_lib = if cfg!(feature = "vulkan") {
        Some(compile_vulkan(&mut cx, &mut cxx))
    } else if cfg!(feature = "cuda") {
        Some(compile_cuda(&mut cx, &mut cxx, featless_cxx))
    } else if cfg!(feature = "opencl") {
        compile_opencl(&mut cx, &mut cxx);
        None
    } else if cfg!(feature = "openblas") {
        compile_openblas(&mut cx);
        None
    } else if cfg!(feature = "blis") {
        compile_blis(&mut cx);
        None
    } else if cfg!(feature = "metal") && cfg!(target_os = "macos") {
        compile_metal(&mut cx, &mut cxx);
        None
    } else if cfg!(feature = "hipblas") {
        Some(compile_hipblas(&mut cx, &mut cxx, featless_cxx))
    } else {
        None
    };

    compile_ggml(cx);
    compile_llama(cxx, &out_path);

    #[cfg(all(
        feature = "compat",
        not(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))
    ))]
    {
        compat::redefine_symbols(out_path, feat_lib);
    }
}

// MacOS will prefix all exported symbols with a leading underscore.
// Additionally, it seems that there are no collision issues when building with both llama and whisper crates, so the
// compat feature can be ignored.

#[cfg(all(
    feature = "compat",
    not(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))
))]
mod compat {
    use std::collections::HashSet;
    use std::fmt::{Display, Formatter};
    use std::process::Command;

    use crate::*;

    pub fn redefine_symbols(out_path: impl AsRef<Path>, additional_lib: Option<&str>) {
        let (ggml_lib_name, llama_lib_name) = lib_names();
        let (nm, objcopy) = tools();
        println!(
            "Modifying {ggml_lib_name} and {llama_lib_name}, symbols acquired via \
        \"{nm}\" and modified via \"{objcopy}\""
        );

        // Modifying symbols exposed by the ggml library

        let out_str = nm_symbols(&nm, ggml_lib_name, &out_path);
        let symbols = get_symbols(
            &out_str,
            [
                Filter {
                    prefix: "ggml",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "ggml",
                    sym_type: 'U',
                },
                Filter {
                    prefix: "ggml",
                    sym_type: 'B',
                },
                Filter {
                    prefix: "gguf",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "quantize",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "dequantize",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "iq2xs",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "iq3xs",
                    sym_type: 'T',
                },
            ],
        );
        objcopy_redefine(&objcopy, ggml_lib_name, PREFIX, symbols, &out_path);

        // Modifying the symbols llama depends on from ggml

        let out_str = nm_symbols(&nm, llama_lib_name, &out_path);
        let symbols = get_symbols(
            &out_str,
            [
                Filter {
                    prefix: "ggml",
                    sym_type: 'U',
                },
                Filter {
                    prefix: "gguf",
                    sym_type: 'U',
                },
            ],
        );
        objcopy_redefine(&objcopy, llama_lib_name, PREFIX, symbols, &out_path);

        if let Some(gpu_lib_name) = additional_lib {
            // Modifying the symbols of the GPU library

            let lib_name = if cfg!(target_family = "windows") {
                format!("{gpu_lib_name}.lib")
            } else if cfg!(target_family = "unix") {
                format!("lib{gpu_lib_name}.a")
            } else {
                println!("cargo:warning=Unknown target family, defaulting to Unix lib names");
                format!("lib{gpu_lib_name}.a")
            };

            let out_str = nm_symbols(&nm, &lib_name, &out_path);
            let symbols = get_symbols(
                &out_str,
                [
                    Filter {
                        prefix: "ggml",
                        sym_type: 'U',
                    },
                    Filter {
                        prefix: "ggml",
                        sym_type: 'T',
                    },
                ],
            );
            objcopy_redefine(&objcopy, &lib_name, PREFIX, symbols, &out_path);
        }
    }

    /// Returns *GGML*'s and *Llama.cpp*'s compiled library names, based on the operating system.
    fn lib_names() -> (&'static str, &'static str) {
        let ggml_lib_name;
        let llama_lib_name;
        if cfg!(target_family = "windows") {
            ggml_lib_name = "ggml.lib";
            llama_lib_name = "llama.lib";
        } else if cfg!(target_family = "unix") {
            ggml_lib_name = "libggml.a";
            llama_lib_name = "libllama.a";
        } else {
            println!("cargo:warning=Unknown target family, defaulting to Unix lib names");
            ggml_lib_name = "libggml.a";
            llama_lib_name = "libllama.a";
        };

        (ggml_lib_name, llama_lib_name)
    }

    /// Returns [`Tool`]s equivalent to [nm][nm] and [objcopy][objcopy].
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn tools() -> (Tool, Tool) {
        let nm_names;
        let objcopy_names;
        let nm_help;
        let objcopy_help;
        if cfg!(target_os = "linux") {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["objcopy", "llvm-objcopy"];
            nm_help = vec!["\"nm\" from GNU Binutils", "\"llvm-nm\" from LLVM"];
            objcopy_help = vec![
                "\"objcopy\" from GNU Binutils",
                "\"llvm-objcopy\" from LLVM",
            ];
        } else if cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "dragonfly"
        )) {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
            nm_help = vec!["\"llvm-nm\" from LLVM 17"];
            objcopy_help = vec!["\"llvm-objcopy\" from LLVM 17"];
        } else {
            nm_names = vec!["llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
            nm_help = vec!["\"llvm-nm\" from LLVM 17"];
            objcopy_help = vec!["\"llvm-objcopy\" from LLVM 17"];
        }

        let nm_env = "NM_PATH";
        println!("cargo:rerun-if-env-changed={nm_env}");
        println!("Looking for \"nm\" or an equivalent tool");
        let nm_name = find_tool(&nm_names, nm_env).unwrap_or_else(move || {
            panic_tool_help("nm", nm_env, &nm_help);
            unreachable!("The function above should have panicked")
        });

        let objcopy_env = "OBJCOPY_PATH";
        println!("cargo:rerun-if-env-changed={objcopy_env}");
        println!("Looking for \"objcopy\" or an equivalent tool..");
        let objcopy_name = find_tool(&objcopy_names, objcopy_env).unwrap_or_else(move || {
            panic_tool_help("objcopy", objcopy_env, &objcopy_help);
            unreachable!("The function above should have panicked")
        });

        (nm_name, objcopy_name)
    }

    /// A command line tool name present in `PATH` or its full [`Path`].
    enum Tool {
        /// The name of a tool present in `PATH`.
        Name(&'static str),

        /// The full [`Path`] to a tool.
        FullPath(PathBuf),
    }

    impl Display for Tool {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                Tool::Name(name) => write!(f, "{}", name),
                Tool::FullPath(path) => write!(f, "{}", path.display()),
            }
        }
    }

    /// Returns the first [`Tool`] found in the system `PATH`, given a list of tool names, returning
    /// the first one found and printing its version.
    ///
    /// If a value is present in the provided environment variable name, it will get checked
    /// instead.
    ///
    /// ## Panic
    /// Returns [`Option::None`] if no [`Tool`] is found.
    fn find_tool(names: &[&'static str], env: &str) -> Option<Tool> {
        if let Ok(path_str) = env::var(env) {
            let path_str = path_str.trim_matches([' ', '"', '\''].as_slice());
            println!("{env} is set, checking if \"{path_str}\" is a valid tool");
            let path = PathBuf::from(&path_str);

            if !path.is_file() {
                panic!("\"{path_str}\" is not a file path.")
            }

            let output = Command::new(path_str)
                .arg("--version")
                .output()
                .unwrap_or_else(|e| panic!("Failed to run \"{path_str} --version\". ({e})"));

            if output.status.success() {
                let out_str = String::from_utf8_lossy(&output.stdout);
                println!("Valid tool found:\n{out_str}");
            } else {
                println!("cargo:warning=Tool \"{path_str}\" found, but could not execute \"{path_str} --version\"")
            }

            return Some(Tool::FullPath(path));
        }

        println!("{env} not set, looking for {names:?} in PATH");
        for name in names {
            if let Ok(output) = Command::new(name).arg("--version").output() {
                if output.status.success() {
                    let out_str = String::from_utf8_lossy(&output.stdout);
                    println!("Valid tool found:\n{out_str}");
                    return Some(Tool::Name(name));
                }
            }
        }

        None
    }

    /// Always panics, printing suggestions for finding the specified tool.
    fn panic_tool_help(name: &str, env: &str, suggestions: &[&str]) {
        let suggestions_str = if suggestions.is_empty() {
            String::new()
        } else {
            let mut suggestions_str = "For your Operating System we recommend:\n".to_string();
            for suggestion in &suggestions[..suggestions.len() - 1] {
                suggestions_str.push_str(&format!("{suggestion}\nOR\n"));
            }
            suggestions_str.push_str(suggestions[suggestions.len() - 1]);
            suggestions_str
        };

        panic!("No suitable tool equivalent to \"{name}\" has been found in PATH, if one is already installed, either add its directory to PATH or set {env} to its full path. {suggestions_str}")
    }

    /// Executes [nm][nm] or an equivalent tool in portable mode and returns the output.
    ///
    /// ## Panic
    /// Will panic on any errors.
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    fn nm_symbols(tool: &Tool, target_lib: &str, out_path: impl AsRef<Path>) -> String {
        let output = Command::new(tool.to_string())
            .current_dir(&out_path)
            .arg(target_lib)
            .args(["-p", "-P"])
            .output()
            .unwrap_or_else(move |e| panic!("Failed to run \"{tool}\". ({e})"));

        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library \"{target_lib}\" ({}):\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Executes [objcopy][objcopy], adding a prefix to the specified symbols of the target library.
    ///
    /// ## Panic
    /// Will panic on any errors.
    ///
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn objcopy_redefine(
        tool: &Tool,
        target_lib: &str,
        prefix: &str,
        symbols: HashSet<&str>,
        out_path: impl AsRef<Path>,
    ) {
        let mut cmd = Command::new(tool.to_string());
        cmd.current_dir(&out_path);
        for symbol in symbols {
            cmd.arg(format!("--redefine-sym={symbol}={prefix}{symbol}"));
        }

        let output = cmd
            .arg(target_lib)
            .output()
            .unwrap_or_else(move |e| panic!("Failed to run \"{tool}\". ({e})"));

        if !output.status.success() {
            panic!(
                "An error has occurred while redefining symbols from library file \"{target_lib}\" ({}):\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    /// A filter for a symbol in a library.
    struct Filter<'a> {
        prefix: &'a str,
        sym_type: char,
    }

    /// Turns **`nm`**'s output into an iterator of [`str`] symbols.
    ///
    /// This function expects **`nm`** to be called using the **`-p`** and **`-P`** flags.
    fn get_symbols<'a, const N: usize>(
        nm_output: &'a str,
        filters: [Filter<'a>; N],
    ) -> HashSet<&'a str> {
        let iter = nm_output
            .lines()
            .map(|symbol| {
                // Strip irrelevant information

                let mut stripped = symbol;
                while stripped.split(' ').count() > 2 {
                    // SAFETY: We just made sure ' ' is present above
                    let idx = unsafe { stripped.rfind(' ').unwrap_unchecked() };
                    stripped = &stripped[..idx]
                }
                stripped
            })
            .filter(move |symbol| {
                // Filter matching symbols

                if symbol.split(' ').count() == 2 {
                    for filter in &filters {
                        if symbol.ends_with(filter.sym_type) && symbol.starts_with(filter.prefix) {
                            return true;
                        }
                    }
                }
                false
            })
            .map(|symbol| &symbol[..symbol.len() - 2]); // Strip the type, so only the symbol remains

        // Filter duplicates
        HashSet::from_iter(iter)
    }
}
