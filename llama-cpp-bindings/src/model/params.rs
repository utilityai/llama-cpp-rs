use crate::LlamaCppError;
use crate::context::params::LlamaContextParams;
use crate::error::{FitError, ModelParamsError};
use crate::model::llama_split_mode_parse_error::LlamaSplitModeParseError;
use crate::model::params::fit_result::FitResult;
use crate::model::params::kv_overrides::KvOverrides;
use crate::model::split_mode::LlamaSplitMode;
use std::ffi::{CStr, c_char};
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr::null;

pub mod fit_result;
pub mod kv_override_value_iterator;
pub mod kv_overrides;
pub mod param_override_value;
pub mod unknown_kv_override_tag;

pub const LLAMA_CPP_MAX_DEVICES: usize = 16;

pub struct LlamaModelParams {
    pub params: llama_cpp_bindings_sys::llama_model_params,
    kv_overrides: Vec<llama_cpp_bindings_sys::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_bindings_sys::llama_model_tensor_buft_override>,
    devices: Pin<Box<[llama_cpp_bindings_sys::ggml_backend_dev_t; LLAMA_CPP_MAX_DEVICES]>>,
    tensor_split: Vec<f32>,
}

impl Debug for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.params.use_mmap)
            .field("use_mlock", &self.params.use_mlock)
            .field("split_mode", &self.split_mode())
            .field("devices", &self.devices)
            .field("kv_overrides", &"vec of kv_overrides")
            .finish_non_exhaustive()
    }
}

impl LlamaModelParams {
    #[must_use]
    pub const fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(self)
    }

    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    ///
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: param_override_value::ParamOverrideValue,
    ) -> Result<(), ModelParamsError> {
        let kv_override = self
            .kv_overrides
            .get_mut(0)
            .ok_or(ModelParamsError::NoAvailableSlot)?;

        if kv_override.key[0] != 0 {
            return Err(ModelParamsError::SlotNotEmpty);
        }

        for (i, &byte) in key.to_bytes_with_nul().iter().enumerate() {
            kv_override.key[i] = c_char::try_from(byte).map_err(|convert_error| {
                ModelParamsError::InvalidCharacterInKey {
                    byte,
                    reason: convert_error.to_string(),
                }
            })?;
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        self.push_kv_override_terminator();

        Ok(())
    }

    fn push_kv_override_terminator(mut self: Pin<&mut Self>) {
        self.params.kv_overrides = null();

        self.kv_overrides
            .push(llama_cpp_bindings_sys::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            });

        self.params.kv_overrides = self.kv_overrides.as_ptr();
    }
}

impl LlamaModelParams {
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) -> Result<(), ModelParamsError> {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps")
    }

    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_buft_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
    ) -> Result<(), ModelParamsError> {
        let buft_override = self
            .buft_overrides
            .get_mut(0)
            .ok_or(ModelParamsError::NoAvailableSlot)?;

        if !buft_override.pattern.is_null() {
            return Err(ModelParamsError::SlotNotEmpty);
        }

        for &byte in key.to_bytes_with_nul() {
            c_char::try_from(byte).map_err(|convert_error| {
                ModelParamsError::InvalidCharacterInKey {
                    byte,
                    reason: convert_error.to_string(),
                }
            })?;
        }

        buft_override.pattern = key.as_ptr();
        buft_override.buft = unsafe { llama_cpp_bindings_sys::ggml_backend_cpu_buffer_type() };

        self.push_buft_override_terminator();

        Ok(())
    }

    fn push_buft_override_terminator(mut self: Pin<&mut Self>) {
        self.params.tensor_buft_overrides = null();

        self.buft_overrides
            .push(llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: std::ptr::null_mut(),
            });

        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();
    }
}

impl LlamaModelParams {
    #[must_use]
    pub const fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    #[must_use]
    pub const fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    #[must_use]
    pub const fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    #[must_use]
    pub const fn use_mmap(&self) -> bool {
        self.params.use_mmap
    }

    #[must_use]
    pub const fn use_mlock(&self) -> bool {
        self.params.use_mlock
    }

    /// # Errors
    /// Returns `LlamaSplitModeParseError` if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, LlamaSplitModeParseError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    #[must_use]
    pub fn devices(&self) -> Vec<usize> {
        let mut backend_devices = Vec::new();
        for i in 0..unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() } {
            let dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(i) };
            backend_devices.push(dev);
        }
        let mut devices = Vec::new();
        for &dev in self.devices.iter() {
            if dev.is_null() {
                break;
            }
            let matched_index = backend_devices
                .iter()
                .enumerate()
                .find(|&(_i, &d)| d == dev)
                .map(|(index, _)| index);

            if let Some(index) = matched_index {
                devices.push(index);
            }
        }
        devices
    }

    #[must_use]
    pub const fn with_n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    #[must_use]
    pub const fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    #[must_use]
    pub const fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    #[must_use]
    pub const fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    #[must_use]
    pub const fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    #[must_use]
    pub const fn with_no_alloc(mut self, no_alloc: bool) -> Self {
        self.params.no_alloc = no_alloc;
        if no_alloc {
            self.params.use_mmap = false;
        }
        self
    }

    #[must_use]
    pub const fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    /// # Errors
    /// Returns `LlamaCppError::BackendDeviceNotFound` if any device index is invalid.
    pub fn with_devices(mut self, devices: &[usize]) -> Result<Self, LlamaCppError> {
        for dev in self.devices.iter_mut() {
            *dev = std::ptr::null_mut();
        }
        let max_devices = crate::max_devices().min(LLAMA_CPP_MAX_DEVICES);
        if devices.len() > max_devices {
            return Err(LlamaCppError::MaxDevicesExceeded(max_devices));
        }
        for (i, &dev) in devices.iter().enumerate() {
            if dev >= unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() } {
                return Err(LlamaCppError::BackendDeviceNotFound(dev));
            }
            let backend_dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(dev) };
            self.devices[i] = backend_dev;
        }
        self.params.devices = self.devices.as_mut_ptr();

        Ok(self)
    }
}

fn fit_params_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_fit_params_status,
    out_unrecognized_status_code: i32,
    out_error: *mut c_char,
) -> Result<(), FitError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_FAILURE => {
            Err(FitError::NoFittingMemoryLayout)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_ERROR => {
            Err(FitError::Aborted)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_RETURNED_UNRECOGNIZED_STATUS_CODE => {
            Err(FitError::UnknownStatus {
                code: out_unrecognized_status_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(FitError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(FitError::Reported { message })
        }
        other => unreachable!("llama_rs_fit_params returned unrecognized wrapper status: {other}"),
    }
}

impl LlamaModelParams {
    /// # Errors
    ///
    /// Returns one of the [`FitError`] variants matching the vendored wrapper's status code.
    pub fn fit_params(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        context_params: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
        log_level: llama_cpp_bindings_sys::ggml_log_level,
    ) -> Result<FitResult, FitError> {
        let max_devices = unsafe { llama_cpp_bindings_sys::llama_max_devices() };
        let max_buft = unsafe { llama_cpp_bindings_sys::llama_max_tensor_buft_overrides() };

        self.tensor_split.clear();
        self.tensor_split.resize(max_devices, 0.0);

        self.buft_overrides.clear();
        self.buft_overrides.resize(
            max_buft + 1,
            llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: std::ptr::null_mut(),
            },
        );

        self.params.tensor_split = null::<f32>();
        self.params.tensor_buft_overrides = null();

        let mut out_unrecognized_status_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_fit_params(
                model_path.as_ptr(),
                &raw mut self.params,
                &raw mut context_params.context_params,
                self.tensor_split.as_mut_ptr(),
                self.buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                n_ctx_min,
                log_level,
                &raw mut out_unrecognized_status_code,
                &raw mut out_error,
            )
        };

        fit_params_status_to_result(status, out_unrecognized_status_code, out_error)?;

        self.params.tensor_split = self.tensor_split.as_ptr();
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        Ok(FitResult {
            n_ctx: context_params.context_params.n_ctx,
        })
    }
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_bindings_sys::llama_model_default_params() };
        Self {
            params: default_params,
            kv_overrides: vec![llama_cpp_bindings_sys::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }],
            buft_overrides: vec![llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: std::ptr::null_mut(),
            }],
            devices: Box::pin([std::ptr::null_mut(); 16]),
            tensor_split: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::split_mode::LlamaSplitMode;

    use super::LlamaModelParams;

    #[test]
    fn default_params_have_expected_values() {
        let params = LlamaModelParams::default();

        assert_eq!(params.n_gpu_layers(), -1);
        assert_eq!(params.main_gpu(), 0);
        assert!(!params.vocab_only());
        assert!(params.use_mmap());
        assert!(!params.use_mlock());
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer));
        assert!(params.devices().is_empty());
    }

    #[test]
    fn with_n_gpu_layers_sets_the_offload_count() {
        let params = LlamaModelParams::default().with_n_gpu_layers(999);

        assert_eq!(params.n_gpu_layers(), 999);
    }

    #[test]
    fn with_n_gpu_layers_sets_value() {
        let params = LlamaModelParams::default().with_n_gpu_layers(32);

        assert_eq!(params.n_gpu_layers(), 32);
    }

    #[test]
    fn with_main_gpu_sets_value() {
        let params = LlamaModelParams::default().with_main_gpu(2);

        assert_eq!(params.main_gpu(), 2);
    }

    #[test]
    fn with_split_mode_none() {
        let params = LlamaModelParams::default().with_split_mode(LlamaSplitMode::None);

        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::None));
    }

    #[test]
    fn with_split_mode_row() {
        let params = LlamaModelParams::default().with_split_mode(LlamaSplitMode::Row);

        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Row));
    }

    #[test]
    fn with_vocab_only_enables() {
        let params = LlamaModelParams::default().with_vocab_only(true);

        assert!(params.vocab_only());
    }

    #[test]
    fn with_vocab_only_disables() {
        let params = LlamaModelParams::default().with_vocab_only(false);

        assert!(!params.vocab_only());
    }

    #[test]
    fn with_use_mmap_enables() {
        let params = LlamaModelParams::default().with_use_mmap(true);

        assert!(params.use_mmap());
    }

    #[test]
    fn with_use_mmap_disables() {
        let params = LlamaModelParams::default().with_use_mmap(false);

        assert!(!params.use_mmap());
    }

    #[test]
    fn with_no_alloc_enables() {
        let params = LlamaModelParams::default().with_no_alloc(true);

        assert!(params.no_alloc());
    }

    #[test]
    fn with_no_alloc_disables() {
        let params = LlamaModelParams::default().with_no_alloc(false);

        assert!(!params.no_alloc());
    }

    #[test]
    fn with_no_alloc_true_disables_mmap() {
        let params = LlamaModelParams::default()
            .with_use_mmap(true)
            .with_no_alloc(true);

        assert!(params.no_alloc());
        assert!(!params.use_mmap());
    }

    #[test]
    fn default_no_alloc_is_false() {
        let params = LlamaModelParams::default();

        assert!(!params.no_alloc());
    }

    #[test]
    fn with_use_mlock_enables() {
        let params = LlamaModelParams::default().with_use_mlock(true);

        assert!(params.use_mlock());
    }

    #[test]
    fn with_use_mlock_disables() {
        let params = LlamaModelParams::default().with_use_mlock(false);

        assert!(!params.use_mlock());
    }

    #[test]
    fn debug_format_contains_field_names() {
        let params = LlamaModelParams::default();
        let debug_output = format!("{params:?}");

        assert!(debug_output.contains("n_gpu_layers"));
        assert!(debug_output.contains("main_gpu"));
        assert!(debug_output.contains("vocab_only"));
        assert!(debug_output.contains("use_mmap"));
        assert!(debug_output.contains("use_mlock"));
        assert!(debug_output.contains("split_mode"));
    }

    #[test]
    fn builder_chaining_preserves_all_values() {
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(10)
            .with_main_gpu(1)
            .with_split_mode(LlamaSplitMode::Row)
            .with_vocab_only(true)
            .with_use_mlock(true);

        assert_eq!(params.n_gpu_layers(), 10);
        assert_eq!(params.main_gpu(), 1);
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Row));
        assert!(params.vocab_only());
        assert!(params.use_mlock());
    }

    #[test]
    fn with_devices_empty_list_succeeds() {
        let params = LlamaModelParams::default().with_devices(&[]);

        assert!(params.is_ok());
        assert!(params.unwrap().devices().is_empty());
    }

    #[test]
    fn with_devices_invalid_index_returns_error() {
        let result = LlamaModelParams::default().with_devices(&[999_999]);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&crate::LlamaCppError::BackendDeviceNotFound(0)),
        );
    }

    #[test]
    fn add_cpu_buft_override_succeeds() {
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let result = params.as_mut().add_cpu_buft_override(c"test_pattern");

        assert!(result.is_ok());
    }

    #[test]
    fn add_cpu_buft_override_twice_fails_with_slot_not_empty() {
        let mut params = std::pin::pin!(LlamaModelParams::default());
        params
            .as_mut()
            .add_cpu_buft_override(c"first_pattern")
            .unwrap();
        let result = params.as_mut().add_cpu_buft_override(c"second_pattern");

        assert_eq!(
            result.unwrap_err(),
            crate::error::ModelParamsError::SlotNotEmpty
        );
    }

    #[test]
    fn add_cpu_moe_override_succeeds() {
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let result = params.as_mut().add_cpu_moe_override();

        assert!(result.is_ok());
    }

    #[test]
    fn append_kv_override_twice_fails_with_slot_not_empty() {
        use crate::model::params::param_override_value::ParamOverrideValue;
        use std::ffi::CString;

        let mut params = std::pin::pin!(LlamaModelParams::default());
        let key = CString::new("first_key").unwrap();
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(1))
            .unwrap();

        let key2 = CString::new("second_key").unwrap();
        let result = params
            .as_mut()
            .append_kv_override(&key2, ParamOverrideValue::Int(2));

        assert_eq!(
            result.unwrap_err(),
            crate::error::ModelParamsError::SlotNotEmpty
        );
    }

    #[test]
    fn with_devices_too_many_returns_max_exceeded() {
        let too_many: Vec<usize> = (0..17).collect();
        let result = LlamaModelParams::default().with_devices(&too_many);

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Max devices exceeded")
        );
    }

    #[test]
    fn with_devices_sets_devices_when_available() {
        #[cfg(feature = "dynamic-backends")]
        crate::load_backends::load_backends().unwrap();

        let dev_count = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() };
        assert!(dev_count > 0, "Test requires at least one backend device");

        let params = LlamaModelParams::default().with_devices(&[0]).unwrap();

        assert_eq!(params.devices().len(), 1);
        assert_eq!(params.devices()[0], 0);
    }

    #[test]
    fn with_devices_invalid_index_returns_not_found() {
        let invalid_index = usize::MAX;
        let result = LlamaModelParams::default().with_devices(&[invalid_index]);

        assert!(result.unwrap_err().to_string().contains("Backend device"));
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn append_kv_override_with_high_byte_returns_invalid_character_error() {
        use crate::model::params::param_override_value::ParamOverrideValue;

        let key_bytes: &[u8] = b"\xff\0";
        let key = std::ffi::CStr::from_bytes_with_nul(key_bytes).unwrap();
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let result = params
            .as_mut()
            .append_kv_override(key, ParamOverrideValue::Int(1));

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&crate::error::ModelParamsError::InvalidCharacterInKey {
                byte: 0,
                reason: String::new(),
            }),
        );
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn add_cpu_buft_override_with_high_byte_returns_invalid_character_error() {
        let key_bytes: &[u8] = b"\xff\0";
        let key = std::ffi::CStr::from_bytes_with_nul(key_bytes).unwrap();
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let result = params.as_mut().add_cpu_buft_override(key);

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&crate::error::ModelParamsError::InvalidCharacterInKey {
                byte: 0,
                reason: String::new(),
            }),
        );
    }

    #[test]
    fn append_kv_override_with_empty_slot_vector_returns_no_available_slot() {
        use crate::model::params::param_override_value::ParamOverrideValue;

        let mut params = LlamaModelParams::default();
        params.kv_overrides.clear();
        let mut pinned = std::pin::pin!(params);

        let result = pinned
            .as_mut()
            .append_kv_override(c"any_key", ParamOverrideValue::Int(1));

        assert_eq!(
            result.unwrap_err(),
            crate::error::ModelParamsError::NoAvailableSlot
        );
    }

    #[test]
    fn add_cpu_buft_override_with_empty_slot_vector_returns_no_available_slot() {
        let mut params = LlamaModelParams::default();
        params.buft_overrides.clear();
        let mut pinned = std::pin::pin!(params);

        let result = pinned.as_mut().add_cpu_buft_override(c"any_pattern");

        assert_eq!(
            result.unwrap_err(),
            crate::error::ModelParamsError::NoAvailableSlot
        );
    }

    #[test]
    #[serial_test::serial]
    fn fit_params_invalid_model_path_returns_error() {
        use crate::context::params::LlamaContextParams;
        use crate::error::FitError;
        use crate::llama_backend::LlamaBackend;

        let _backend = LlamaBackend::init();
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let mut context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; crate::max_devices()];

        let bogus_path = c"/nonexistent/path/to/model.gguf";
        let result = params.as_mut().fit_params(
            bogus_path,
            &mut context_params,
            &mut margins,
            512,
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
        );

        assert!(
            matches!(result, Err(FitError::Aborted | FitError::Reported { .. })),
            "expected Aborted or Reported, got {result:?}"
        );
    }

    #[test]
    fn fit_params_status_ok_returns_ok() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_OK,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn fit_params_status_reported_failure_returns_no_fitting_memory_layout() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_FAILURE,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(crate::error::FitError::NoFittingMemoryLayout));
    }

    #[test]
    fn fit_params_status_reported_error_returns_aborted() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_ERROR,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(crate::error::FitError::Aborted));
    }

    #[test]
    fn fit_params_status_unrecognized_code_returns_unknown_status() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_RETURNED_UNRECOGNIZED_STATUS_CODE,
            42,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::error::FitError::UnknownStatus { code: 42 })
        );
    }

    #[test]
    fn fit_params_status_allocation_failed_returns_not_enough_memory() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(crate::error::FitError::NotEnoughMemory));
    }

    #[test]
    fn fit_params_status_cxx_exception_returns_reported_with_unknown_error() {
        let result = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_THREW_CXX_EXCEPTION,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::error::FitError::Reported {
                message: "unknown error".to_owned()
            })
        );
    }

    #[test]
    #[should_panic(expected = "unrecognized wrapper status")]
    fn fit_params_status_out_of_range_panics() {
        let _ = super::fit_params_status_to_result(
            llama_cpp_bindings_sys::llama_rs_fit_params_status::MAX,
            0,
            std::ptr::null_mut(),
        );
    }
}
