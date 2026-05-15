use anyhow::Result;
#[cfg(any(
    test,
    feature = "cuda",
    feature = "cuda-no-vmm",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm",
))]
use llama_cpp_bindings::llama_backend_device::LlamaBackendDevice;
use llama_cpp_bindings::llama_backend_device::list_llama_ggml_backend_devices;
use llama_cpp_bindings::model::params::LlamaModelParams;

#[must_use]
pub fn inference_model_params() -> LlamaModelParams {
    let params = LlamaModelParams::default();

    #[cfg(any(
        feature = "cuda",
        feature = "cuda-no-vmm",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm",
    ))]
    let params = params.with_n_gpu_layers(999);

    params
}

/// Confirms every compile-time backend feature has a matching ggml backend registered at runtime.
///
/// Always asserts at least the CPU backend is registered (any llama.cpp build registers it);
/// when a GPU backend feature is enabled, also asserts the corresponding GPU backend is present.
///
/// # Errors
///
/// Returns an error when no ggml backends are registered, or when a compiled-in GPU backend
/// feature has no matching device. The error message names the missing backend(s) and lists
/// the backends that *are* registered, so misconfiguration is easy to diagnose.
pub fn require_compiled_backends_present() -> Result<()> {
    let devices = list_llama_ggml_backend_devices();

    if devices.is_empty() {
        anyhow::bail!("no ggml backends registered; even CPU-only builds register a CPU backend");
    }

    #[cfg(feature = "cuda")]
    require_backend(&devices, "cuda", &["CUDA"])?;
    #[cfg(feature = "cuda-no-vmm")]
    require_backend(&devices, "cuda-no-vmm", &["CUDA"])?;
    #[cfg(feature = "metal")]
    require_backend(&devices, "metal", &["Metal", "MTL"])?;
    #[cfg(feature = "vulkan")]
    require_backend(&devices, "vulkan", &["Vulkan"])?;
    #[cfg(feature = "rocm")]
    require_backend(&devices, "rocm", &["HIP", "ROCm"])?;

    Ok(())
}

#[cfg(any(
    test,
    feature = "cuda",
    feature = "cuda-no-vmm",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm",
))]
fn require_backend(
    devices: &[LlamaBackendDevice],
    feature: &str,
    accepted_names: &[&str],
) -> Result<()> {
    let found = devices.iter().any(|device| {
        accepted_names
            .iter()
            .any(|wanted| device.backend.eq_ignore_ascii_case(wanted))
    });

    if !found {
        let summary: Vec<String> = devices
            .iter()
            .map(|device| format!("{}/{:?}", device.backend, device.device_type))
            .collect();

        anyhow::bail!(
            "feature `{feature}` enabled but no matching backend ({}) is registered; available: [{}]",
            accepted_names.join(" / "),
            summary.join(", ")
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::llama_backend_device::LlamaBackendDevice;
    use llama_cpp_bindings::llama_backend_device_type::LlamaBackendDeviceType;

    use super::require_backend;

    fn synthetic_device(backend: &str, device_type: LlamaBackendDeviceType) -> LlamaBackendDevice {
        LlamaBackendDevice {
            index: 0,
            name: format!("{backend}0"),
            description: "synthetic test device".to_owned(),
            backend: backend.to_owned(),
            memory_total: 0,
            memory_free: 0,
            device_type,
        }
    }

    use anyhow::Result;
    use anyhow::anyhow;

    #[test]
    fn require_backend_succeeds_when_backend_name_matches_case_insensitively() -> Result<()> {
        let devices = vec![synthetic_device("cuda", LlamaBackendDeviceType::Gpu)];

        require_backend(&devices, "cuda", &["CUDA"])
    }

    #[test]
    fn require_backend_succeeds_with_any_of_multiple_accepted_names() -> Result<()> {
        let devices = vec![synthetic_device("HIP", LlamaBackendDeviceType::Gpu)];

        require_backend(&devices, "rocm", &["HIP", "ROCm"])
    }

    #[test]
    fn require_backend_fails_with_message_naming_feature_and_accepted_names_when_missing()
    -> Result<()> {
        let devices = vec![synthetic_device("Vulkan", LlamaBackendDeviceType::Gpu)];

        let error = require_backend(&devices, "cuda", &["CUDA"])
            .err()
            .ok_or_else(|| anyhow!("expected error when CUDA missing"))?;

        let message = format!("{error:#}");

        if !message.contains("`cuda`") {
            return Err(anyhow!("missing feature name: {message}"));
        }
        if !message.contains("CUDA") {
            return Err(anyhow!("missing accepted name: {message}"));
        }
        if !message.contains("Vulkan") {
            return Err(anyhow!("missing actual-backend summary: {message}"));
        }

        Ok(())
    }

    #[test]
    fn require_backend_fails_when_devices_list_is_empty() -> Result<()> {
        let devices: Vec<LlamaBackendDevice> = Vec::new();

        if require_backend(&devices, "metal", &["Metal"]).is_ok() {
            return Err(anyhow!("expected Err for empty device list"));
        }

        Ok(())
    }
}
