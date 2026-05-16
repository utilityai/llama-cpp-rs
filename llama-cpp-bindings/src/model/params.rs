//! A safe wrapper around `llama_model_params`.

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

/// The maximum number of devices supported.
///
/// The real maximum number of devices is the lesser one of this value and the value returned by
/// `llama_cpp_bindings::max_devices()`.
pub const LLAMA_CPP_MAX_DEVICES: usize = 16;

/// A safe wrapper around `llama_model_params`.
pub struct LlamaModelParams {
    /// The underlying `llama_model_params` from the C API.
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
    /// See [`KvOverrides`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = Box::pin(LlamaModelParams::default());
    /// let kv_overrides = params.kv_overrides();
    /// let count = kv_overrides.into_iter().count();
    /// assert_eq!(count, 0);
    /// ```
    #[must_use]
    pub const fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(self)
    }

    /// Appends a key-value override to the model parameters. It must be pinned as this creates a self-referential struct.
    ///
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::ffi::{CStr, CString};
    /// use std::pin::pin;
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// # use llama_cpp_bindings::model::params::param_override_value::ParamOverrideValue;
    /// let mut params = pin!(LlamaModelParams::default());
    /// let key = CString::new("key").expect("CString::new failed");
    /// params.as_mut().append_kv_override(&key, ParamOverrideValue::Int(50)).unwrap();
    ///
    /// let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
    /// assert_eq!(kv_overrides.len(), 1);
    ///
    /// let (k, v) = &kv_overrides[0];
    /// assert_eq!(v, &ParamOverrideValue::Int(50));
    ///
    /// assert_eq!(k.to_bytes(), b"key", "expected key to be 'key', was {:?}", k);
    /// ```
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

    /// Pushes the trailing zero-tag sentinel onto `kv_overrides` and refreshes
    /// `params.kv_overrides`. The cached pointer is nulled before [`Vec::push`]
    /// so that a relocation-induced panic never leaves a dangling pointer in
    /// `params`.
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
    /// Adds buffer type overrides to move all mixture-of-experts layers to CPU.
    ///
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) -> Result<(), ModelParamsError> {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps")
    }

    /// Appends a buffer type override to the model parameters, to move layers matching pattern to CPU.
    /// It must be pinned as this creates a self-referential struct.
    ///
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

    /// Pushes the trailing null-pattern sentinel onto `buft_overrides` and
    /// refreshes `params.tensor_buft_overrides`. The cached pointer is nulled
    /// before [`Vec::push`] so that a relocation-induced panic never leaves a
    /// dangling pointer in `params`.
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
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub const fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub const fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub const fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// use mmap if possible
    #[must_use]
    pub const fn use_mmap(&self) -> bool {
        self.params.use_mmap
    }

    /// force system to keep model in RAM
    #[must_use]
    pub const fn use_mlock(&self) -> bool {
        self.params.use_mlock
    }

    /// get the split mode
    ///
    /// # Errors
    /// Returns `LlamaSplitModeParseError` if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, LlamaSplitModeParseError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    /// get the devices
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

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// sets the main GPU
    ///
    /// To enable this option, you must set `split_mode` to `LlamaSplitMode::None` to enable single GPU mode.
    #[must_use]
    pub const fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub const fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// sets `use_mmap`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default().with_use_mmap(false);
    /// assert!(!params.use_mmap());
    /// ```
    #[must_use]
    pub const fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    /// Get `no_alloc`
    #[must_use]
    pub const fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    /// Set `no_alloc`. When enabled, tensor data is not allocated.
    /// Incompatible with `use_mmap`, so enabling this also disables mmap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default().with_no_alloc(true);
    /// assert!(params.no_alloc());
    /// assert!(!params.use_mmap());
    /// ```
    #[must_use]
    pub const fn with_no_alloc(mut self, no_alloc: bool) -> Self {
        self.params.no_alloc = no_alloc;
        if no_alloc {
            self.params.use_mmap = false;
        }
        self
    }

    /// sets `use_mlock`
    #[must_use]
    pub const fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    /// sets `split_mode`
    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    /// sets `devices`
    ///
    /// The devices are specified as indices that correspond to the ggml backend device indices.
    ///
    /// The maximum number of devices is 16.
    ///
    /// You don't need to specify CPU or ACCEL devices.
    ///
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

impl LlamaModelParams {
    /// Automatically fit model and context parameters to available device memory.
    ///
    /// Wraps llama.cpp's `common_fit_params`. Given a model path, available per-device memory
    /// margins, and a minimum context size, it fills in `n_gpu_layers`, `tensor_split`, and
    /// `tensor_buft_overrides` to fit the model to the available VRAM, and may reduce
    /// `cparams.n_ctx` if needed. On success the model and context params are updated in place.
    ///
    /// # Requirements
    ///
    /// Per the C API docstring, only parameters that still hold their default value are
    /// modified. In practice this means:
    /// - `n_gpu_layers` must be at its default (`-1`). Do not call
    ///   [`with_n_gpu_layers`](Self::with_n_gpu_layers) before this.
    /// - No `tensor_buft_overrides` may be set. Do not call
    ///   [`add_cpu_buft_override`](Self::add_cpu_buft_override) or
    ///   [`add_cpu_moe_override`](Self::add_cpu_moe_override) before this.
    /// - `cparams.n_ctx` is only auto-selected if it is `0`; otherwise it is left alone.
    ///
    /// # Arguments
    ///
    /// - `model_path` — path to the GGUF model file as a C string.
    /// - `context_params` — context parameters; `n_ctx` may be modified (see above).
    /// - `margins` — memory margin per device in bytes. Must have at least
    ///   `crate::max_devices()` elements.
    /// - `n_ctx_min` — minimum context size to preserve when reducing memory usage.
    /// - `log_level` — minimum log level for fitting output; lower levels go to the debug log.
    ///
    /// # Thread safety
    ///
    /// This function is **not** thread safe: the underlying C call mutates the global
    /// llama logger state.
    ///
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

        match status {
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_OK => {}
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_FAILURE => {
                return Err(FitError::VendoredReportedFailure);
            }
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_ERROR => {
                return Err(FitError::VendoredReportedError);
            }
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_RETURNED_UNRECOGNIZED_STATUS_CODE => {
                return Err(FitError::VendoredReturnedUnrecognizedStatusCode {
                    code: out_unrecognized_status_code,
                });
            }
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED => {
                return Err(FitError::ErrorStringAllocationFailed);
            }
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe {
                    crate::ffi_error_reader::read_and_free_cpp_error(out_error)
                };
                return Err(FitError::VendoredThrewCxxException { message });
            }
            other => {
                unreachable!(
                    "llama_rs_fit_params returned unrecognized wrapper status: {other}"
                );
            }
        }

        self.params.tensor_split = self.tensor_split.as_ptr();
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        Ok(FitResult {
            n_ctx: context_params.context_params.n_ctx,
        })
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_bindings::model::params::LlamaModelParams;
/// use llama_cpp_bindings::model::split_mode::LlamaSplitMode;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), -1, "n_gpu_layers should be -1");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer), "split_mode should be LAYER");
/// assert_eq!(params.devices().len(), 0, "devices should be empty");
/// ```
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
    fn n_gpu_layers_overflow_clamps_to_max() {
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);

        assert_eq!(params.n_gpu_layers(), i32::MAX);
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
            result.unwrap_err(),
            crate::LlamaCppError::BackendDeviceNotFound(999_999)
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

        assert!(matches!(
            result,
            Err(crate::error::ModelParamsError::InvalidCharacterInKey { byte: 0xff, .. })
        ));
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn add_cpu_buft_override_with_high_byte_returns_invalid_character_error() {
        let key_bytes: &[u8] = b"\xff\0";
        let key = std::ffi::CStr::from_bytes_with_nul(key_bytes).unwrap();
        let mut params = std::pin::pin!(LlamaModelParams::default());
        let result = params.as_mut().add_cpu_buft_override(key);

        assert!(matches!(
            result,
            Err(crate::error::ModelParamsError::InvalidCharacterInKey { byte: 0xff, .. })
        ));
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
            matches!(
                result,
                Err(FitError::VendoredReportedError | FitError::VendoredThrewCxxException { .. })
            ),
            "expected VendoredReportedError or VendoredThrewCxxException, got {result:?}"
        );
    }
}
