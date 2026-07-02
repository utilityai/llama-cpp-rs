use std::env;
use std::path::Path;
use std::path::PathBuf;

use crate::target_os::TargetOs;

const LLAMA_SUBMODULE_RELATIVE_PATH: &str = "../llama-cpp-bindings-sys/llama.cpp";

const GRAMMAR_SOURCE_FILES: &[&str] = &[
    "src/llama-grammar.cpp",
    "src/llama-vocab.cpp",
    "src/unicode.cpp",
    "src/unicode-data.cpp",
    "src/llama-impl.cpp",
];

pub fn build_gbnf() {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR env var is required");
    let target_triple = env::var("TARGET").expect("TARGET env var is required in build scripts");
    let target_os = TargetOs::from_target_triple(&target_triple)
        .unwrap_or_else(|error| panic!("Failed to parse target OS: {error}"));

    let ggml_include = env::var("DEP_GGML_INCLUDE")
        .expect("DEP_GGML_INCLUDE must be provided by llama-cpp-ggml-sys");
    let ggml_lib =
        env::var("DEP_GGML_LIB").expect("DEP_GGML_LIB must be provided by llama-cpp-ggml-sys");

    let manifest = PathBuf::from(&manifest_dir);
    let llama_src = manifest.join(LLAMA_SUBMODULE_RELATIVE_PATH);
    let grammar_source_dir = llama_src.join("src");

    assert!(
        grammar_source_dir.join("llama-grammar.cpp").is_file(),
        "grammar source not found under {}; ensure the llama.cpp submodule is checked out",
        grammar_source_dir.display()
    );

    register_rebuild_triggers(&manifest);

    compile_wrapper(
        &manifest,
        &llama_src,
        &grammar_source_dir,
        Path::new(&ggml_include),
        &target_os,
    );

    emit_link_directives(Path::new(&ggml_lib), &target_os);
}

fn compile_wrapper(
    manifest: &Path,
    llama_src: &Path,
    grammar_source_dir: &Path,
    ggml_include: &Path,
    target_os: &TargetOs,
) {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .include(manifest)
        .include(llama_src.join("include"))
        .include(grammar_source_dir)
        .include(llama_src.join("ggml/include"))
        .include(ggml_include)
        .flag_if_supported("-std=c++17")
        .pic(true)
        .file(manifest.join("wrapper_gbnf.cpp"));

    for source in GRAMMAR_SOURCE_FILES {
        build.file(llama_src.join(source));
    }

    if target_os.is_msvc() {
        build.flag("/std:c++17");
        build.flag("/EHsc");
    }

    build.compile("llama_cpp_gbnf_wrapper");
}

fn emit_link_directives(ggml_lib: &Path, target_os: &TargetOs) {
    println!("cargo:rustc-link-search=native={}", ggml_lib.display());
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    match target_os {
        TargetOs::Linux | TargetOs::Android => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(_) => {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=dylib=c++");
        }
        TargetOs::Windows(_) => {}
    }
}

fn register_rebuild_triggers(manifest: &Path) {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("wrapper_gbnf.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("wrapper_gbnf.h").display()
    );
}
