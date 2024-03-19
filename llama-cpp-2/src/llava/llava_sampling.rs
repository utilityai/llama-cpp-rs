//! sampling
//!

use std::{ffi::c_int, ptr::NonNull};

use crate::{context::LlamaContext, LlamaContextLoadError};

/// sampling params
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaSamplingParams {
    pub(crate) params: NonNull<llama_cpp_sys_2::llama_sampling_params>,
}

impl LlamaSamplingParams {
    /// default sampling params
    pub fn default() -> Result<Self, LlamaContextLoadError> {
        let params = unsafe { llama_cpp_sys_2::llama_sampling_params_default() };

        let params = NonNull::new(params).ok_or(LlamaContextLoadError::NullReturn)?;
        Ok(Self { params })
    }
}

impl Drop for LlamaSamplingParams {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampling_params_free(self.params.as_ptr());
        }
    }
}

/// sampling context
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaSamplingContext {
    pub(crate) context: NonNull<llama_cpp_sys_2::llama_sampling_context>,
}

impl LlamaSamplingContext {
    /// create sampling context
    pub fn init(params: &LlamaSamplingParams) -> Result<Self, LlamaContextLoadError> {
        let context =
            unsafe { llama_cpp_sys_2::llama_sampling_context_init(params.params.as_ptr()) };

        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(Self { context })
    }
}

impl Drop for LlamaSamplingContext {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_sampling_context_free(self.context.as_ptr());
        }
    }
}

/// sample
pub fn llava_sample(
    ctx_sampling: &mut LlamaSamplingContext,
    ctx_llama: &mut LlamaContext,
    n_past: &mut c_int,
) -> String {
    let result = unsafe {
        let result = llama_cpp_sys_2::llava_sample(
            ctx_sampling.context.as_mut(),
            ctx_llama.context.as_mut(),
            n_past,
        );
        std::ffi::CStr::from_ptr(result)
            .to_string_lossy()
            .into_owned()
    };

    result
}
