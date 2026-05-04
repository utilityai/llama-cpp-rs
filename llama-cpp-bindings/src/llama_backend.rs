//! Representation of an initialized llama backend

use crate::LlamaCppError;
use crate::llama_backend_numa_strategy::NumaStrategy;
use llama_cpp_bindings_sys::ggml_log_level;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

/// Representation of an initialized llama backend.
///
/// This is required as a parameter for most llama functions as the backend must be initialized
/// before any llama functions are called. This type is proof of initialization.
#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    /// Mark the llama backend as initialized
    fn mark_init() -> crate::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => Err(LlamaCppError::BackendAlreadyInitialized),
        }
    }

    /// Initialize the llama backend (without numa).
    ///
    /// # Examples
    ///
    /// ```
    ///# use llama_cpp_bindings::llama_backend::LlamaBackend;
    ///# use llama_cpp_bindings::LlamaCppError;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    ///
    /// let backend = LlamaBackend::init()?;
    /// // the llama backend can only be initialized once
    /// assert_eq!(Err(LlamaCppError::BackendAlreadyInitialized), LlamaBackend::init());
    ///
    ///# Ok(())
    ///# }
    /// ```
    /// # Errors
    /// Returns an error if the backend was already initialized.
    #[tracing::instrument(skip_all)]
    pub fn init() -> crate::Result<Self> {
        Self::mark_init()?;
        unsafe { llama_cpp_bindings_sys::llama_backend_init() }
        Ok(Self {})
    }

    /// Initialize the llama backend (with numa).
    /// ```
    ///# use llama_cpp_bindings::llama_backend::LlamaBackend;
    ///# use std::error::Error;
    ///# use llama_cpp_bindings::llama_backend_numa_strategy::NumaStrategy;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    /// let llama_backend = LlamaBackend::init_numa(NumaStrategy::Mirror)?;
    ///
    ///# Ok(())
    ///# }
    /// ```
    /// # Errors
    /// Returns an error if the backend was already initialized.
    #[tracing::instrument(skip_all)]
    pub fn init_numa(strategy: NumaStrategy) -> crate::Result<Self> {
        Self::mark_init()?;
        unsafe {
            llama_cpp_bindings_sys::llama_numa_init(
                llama_cpp_bindings_sys::ggml_numa_strategy::from(strategy),
            );
        }
        Ok(Self {})
    }

    /// Was the code built for a GPU backend & is a supported one available.
    #[must_use]
    pub fn supports_gpu_offload(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_gpu_offload() }
    }

    /// Does this platform support loading the model via mmap.
    #[must_use]
    pub fn supports_mmap(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
    }

    /// Does this platform support locking the model in RAM.
    #[must_use]
    pub fn supports_mlock(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
    }

    /// Change the output of llama.cpp's logging to be voided instead of pushed to `stderr`.
    pub fn void_logs(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_log_set(Some(void_log), std::ptr::null_mut());
        }
    }
}

const unsafe extern "C" fn void_log(
    _level: ggml_log_level,
    _text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
}

/// Drops the llama backend.
/// ```
///
///# use llama_cpp_bindings::llama_backend::LlamaBackend;
///# use std::error::Error;
///
///# fn main() -> Result<(), Box<dyn Error>> {
/// let backend = LlamaBackend::init()?;
/// drop(backend);
/// // can be initialized again after being dropped
/// let backend = LlamaBackend::init()?;
///# Ok(())
///# }
///
/// ```
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        LLAMA_BACKEND_INITIALIZED.store(false, SeqCst);
        unsafe { llama_cpp_bindings_sys::llama_backend_free() }
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::LlamaBackend;
    use crate::LlamaCppError;

    #[test]
    fn void_log_callback_does_not_panic() {
        unsafe {
            super::void_log(
                llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
                c"test".as_ptr(),
                std::ptr::null_mut(),
            );
        }
    }

    #[test]
    #[serial]
    fn init_succeeds() {
        let backend = LlamaBackend::init();
        assert!(backend.is_ok());
    }

    #[test]
    #[serial]
    fn double_init_returns_error() {
        let _backend = LlamaBackend::init().unwrap();
        let second = LlamaBackend::init();
        assert_eq!(
            second.unwrap_err(),
            LlamaCppError::BackendAlreadyInitialized
        );
    }

    #[test]
    #[serial]
    fn feature_queries_return_bools() {
        let backend = LlamaBackend::init().unwrap();
        let _gpu = backend.supports_gpu_offload();
        let _mmap = backend.supports_mmap();
        let _mlock = backend.supports_mlock();
    }

    #[cfg(feature = "tests_that_use_llms")]
    #[test]
    #[serial]
    fn void_logs_suppresses_output() {
        let mut backend = LlamaBackend::init().unwrap();
        backend.void_logs();
        let model_path = crate::test_model::download_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let _model =
            crate::model::LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();
    }

    #[test]
    #[serial]
    fn drop_and_reinit_works() {
        let backend = LlamaBackend::init().unwrap();
        drop(backend);
        let backend = LlamaBackend::init();
        assert!(backend.is_ok());
    }

    #[test]
    #[serial]
    fn init_numa_succeeds() {
        use crate::llama_backend_numa_strategy::NumaStrategy;

        let backend = LlamaBackend::init_numa(NumaStrategy::Disabled);
        assert!(backend.is_ok());
    }
}
