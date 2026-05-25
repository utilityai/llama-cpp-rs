use std::ffi::c_char;

use crate::llama_backend_device_type::device_type_from_raw;

pub use crate::llama_backend_device_type::LlamaBackendDeviceType;

#[derive(Debug, Clone)]
pub struct LlamaBackendDevice {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub backend: String,
    pub memory_total: usize,
    pub memory_free: usize,
    pub device_type: LlamaBackendDeviceType,
}

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .to_string()
    }
}

#[must_use]
pub fn list_llama_ggml_backend_devices() -> Vec<LlamaBackendDevice> {
    let mut devices = Vec::new();
    let device_count = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() };

    for device_index in 0..device_count {
        let dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(device_index) };
        let props = unsafe {
            let mut props = std::mem::zeroed();
            llama_cpp_bindings_sys::ggml_backend_dev_get_props(dev, &raw mut props);
            props
        };
        let name = cstr_to_string(props.name);
        let description = cstr_to_string(props.description);
        let backend_reg = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_backend_reg(dev) };
        let backend_name = unsafe { llama_cpp_bindings_sys::ggml_backend_reg_name(backend_reg) };
        let backend = cstr_to_string(backend_name);
        let memory_total = props.memory_total;
        let memory_free = props.memory_free;
        let device_type = device_type_from_raw(props.type_);
        devices.push(LlamaBackendDevice {
            index: device_index,
            name,
            description,
            backend,
            memory_total,
            memory_free,
            device_type,
        });
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::{cstr_to_string, list_llama_ggml_backend_devices};

    #[test]
    fn cstr_to_string_with_null_returns_empty() {
        let result = cstr_to_string(std::ptr::null());

        assert_eq!(result, "");
    }

    #[test]
    fn cstr_to_string_with_valid_ptr() {
        let result = cstr_to_string(c"hello".as_ptr());

        assert_eq!(result, "hello");
    }

    #[test]
    fn list_devices_returns_at_least_one() {
        #[cfg(feature = "dynamic-backends")]
        crate::load_backends::load_backends().unwrap();

        let devices = list_llama_ggml_backend_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].index, 0);
        assert!(!devices[0].name.is_empty());
    }
}
