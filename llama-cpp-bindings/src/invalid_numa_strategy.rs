#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidNumaStrategy(pub llama_cpp_bindings_sys::ggml_numa_strategy);
