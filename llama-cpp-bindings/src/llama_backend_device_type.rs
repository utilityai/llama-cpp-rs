#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaBackendDeviceType {
    Cpu,
    Accelerator,
    Gpu,
    IntegratedGpu,
    Unknown,
}

#[must_use]
pub const fn device_type_from_raw(
    raw_type: llama_cpp_bindings_sys::ggml_backend_dev_type,
) -> LlamaBackendDeviceType {
    match raw_type {
        llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU => LlamaBackendDeviceType::Cpu,
        llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL => {
            LlamaBackendDeviceType::Accelerator
        }
        llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU => LlamaBackendDeviceType::Gpu,
        llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU => {
            LlamaBackendDeviceType::IntegratedGpu
        }
        _ => LlamaBackendDeviceType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaBackendDeviceType;
    use super::device_type_from_raw;

    #[test]
    fn device_type_from_raw_all_variants() {
        assert_eq!(
            device_type_from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU),
            LlamaBackendDeviceType::Cpu
        );
        assert_eq!(
            device_type_from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL),
            LlamaBackendDeviceType::Accelerator
        );
        assert_eq!(
            device_type_from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU),
            LlamaBackendDeviceType::Gpu
        );
        assert_eq!(
            device_type_from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU),
            LlamaBackendDeviceType::IntegratedGpu
        );
        assert_eq!(device_type_from_raw(9999), LlamaBackendDeviceType::Unknown);
    }
}
