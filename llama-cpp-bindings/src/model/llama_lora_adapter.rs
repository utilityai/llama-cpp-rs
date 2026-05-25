use std::ptr::NonNull;

#[derive(Debug)]
#[repr(transparent)]
pub struct LlamaLoraAdapter {
    pub lora_adapter: NonNull<llama_cpp_bindings_sys::llama_adapter_lora>,
}
