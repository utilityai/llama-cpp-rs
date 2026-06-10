#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSplitModeParseError {
    pub value: llama_cpp_bindings_sys::llama_split_mode,
    pub context: String,
}
