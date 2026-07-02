use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

use llama_cpp_bindings_sys::ggml_log_level;

use crate::error::llama_cpp_error::LlamaCppError;
use crate::llama_backend_numa_strategy::NumaStrategy;

#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    fn mark_init() -> crate::error::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_was_uninitialized) => Ok(()),
            Err(_was_already_initialized) => Err(LlamaCppError::BackendAlreadyInitialized),
        }
    }

    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init() -> crate::error::Result<Self> {
        Self::mark_init()?;
        unsafe { llama_cpp_bindings_sys::llama_backend_init() }
        Ok(Self {})
    }

    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init_numa(strategy: NumaStrategy) -> crate::error::Result<Self> {
        Self::mark_init()?;
        unsafe {
            llama_cpp_bindings_sys::llama_numa_init(
                llama_cpp_bindings_sys::ggml_numa_strategy::from(strategy),
            );
        }
        Ok(Self {})
    }

    #[must_use]
    pub fn supports_gpu_offload(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_gpu_offload() }
    }

    #[must_use]
    pub fn supports_mmap(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
    }

    #[must_use]
    pub fn supports_mlock(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
    }

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
    use crate::error::llama_cpp_error::LlamaCppError;

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
        let second_err = LlamaBackend::init().unwrap_err();

        assert_eq!(
            std::mem::discriminant(&second_err),
            std::mem::discriminant(&LlamaCppError::BackendAlreadyInitialized),
            "expected BackendAlreadyInitialized, got {second_err:?}"
        );
    }

    #[test]
    #[serial]
    fn init_numa_returns_error_when_backend_already_initialized() {
        use crate::llama_backend_numa_strategy::NumaStrategy;

        let _backend = LlamaBackend::init().unwrap();
        let second_err = LlamaBackend::init_numa(NumaStrategy::Disabled).unwrap_err();

        assert_eq!(
            std::mem::discriminant(&second_err),
            std::mem::discriminant(&LlamaCppError::BackendAlreadyInitialized),
            "expected BackendAlreadyInitialized, got {second_err:?}"
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
