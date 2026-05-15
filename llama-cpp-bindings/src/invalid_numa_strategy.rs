/// An invalid numa strategy was provided.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidNumaStrategy(
    /// The invalid numa strategy that was provided.
    pub llama_cpp_bindings_sys::ggml_numa_strategy,
);
