use std::path::Path;

use crate::target_os::TargetOs;

pub fn compile_cpp_wrappers(llama_src: &Path, target_os: &TargetOs) {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .file("wrapper_chat_parse.cpp")
        .file("wrapper_common.cpp")
        .file("wrapper_fit.cpp")
        .file("wrapper_reasoning.cpp")
        .file("wrapper_tool_calls.cpp")
        .file("marker_probes/chunked_thinking.cpp")
        .file("marker_probes/registry.cpp")
        .include(".")
        .include(llama_src)
        .include(llama_src.join("common"))
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    if target_os.is_msvc() {
        build.flag("/std:c++17");
    }

    if target_os.is_android() && cfg!(feature = "static-stdcxx") {
        build.cpp_link_stdlib(None);
    }

    build.compile("llama_cpp_bindings_sys_common_wrapper");
}
