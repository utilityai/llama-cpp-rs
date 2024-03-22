//! safe wrapper for multimodel model llava
//! CLIP(Contrastive Language–Image Pre-training)

use std::{
    ffi::{c_int, CString},
    path::Path,
    ptr::NonNull,
};

use crate::{llama_backend::LlamaBackend, LlamaModelLoadError};

/// A safe wrapper around `clip_ctx`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipCtx {
    pub(crate) ctx: NonNull<llama_cpp_sys_2::clip_ctx>,
}

impl ClipCtx {
    /// load clip context from a path
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        path: impl AsRef<Path>,
        verbosity: c_int,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let clip_ctx = unsafe { llama_cpp_sys_2::clip_model_load(cstr.as_ptr(), verbosity) };

        let ctx = NonNull::new(clip_ctx).ok_or(LlamaModelLoadError::NullResult)?;
        tracing::debug!(?path, "Loaded model");
        Ok(ClipCtx { ctx })
    }

    /// load CPU-only clip context from a path
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_cpu_from_file(
        path: impl AsRef<Path>,
        verbosity: c_int,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let clip_ctx = unsafe { llama_cpp_sys_2::clip_model_load_cpu(cstr.as_ptr(), verbosity) };

        let ctx = NonNull::new(clip_ctx).ok_or(LlamaModelLoadError::NullResult)?;
        tracing::debug!(?path, "Loaded model for CPU");
        Ok(ClipCtx { ctx })
    }

    /// get the number of bytes in the embedding
    pub fn embd_nbytes(&self) -> usize {
        unsafe { llama_cpp_sys_2::clip_embd_nbytes(self.ctx.as_ptr()) }
    }

    /// get size of image
    pub fn image_size(&self) -> i32 {
        unsafe { llama_cpp_sys_2::clip_image_size(self.ctx.as_ptr()) }
    }

    /// get size of patch
    pub fn patch_size(&self) -> i32 {
        unsafe { llama_cpp_sys_2::clip_patch_size(self.ctx.as_ptr()) }
    }

    /// get size of hidden
    pub fn hidden_size(&self) -> i32 {
        unsafe { llama_cpp_sys_2::clip_hidden_size(self.ctx.as_ptr()) }
    }

    /// patch merge type
    /// TODO: should be enum, not string
    pub fn patch_merge_type(&self) -> String {
        let cstr = unsafe {
            let cstr = llama_cpp_sys_2::clip_patch_merge_type(self.ctx.as_ptr());
            std::ffi::CStr::from_ptr(cstr)
        };
        cstr.to_string_lossy().into_owned()
    }

    /// image grid
    pub fn image_grid(&self) -> &[i32] {
        unsafe {
            let grid = llama_cpp_sys_2::clip_image_grid(self.ctx.as_ptr());
            let grid = std::slice::from_raw_parts(grid, 32);
            grid
        }
    }

    /// n patchs
    pub fn n_patches(&self) -> c_int {
        unsafe { llama_cpp_sys_2::clip_n_patches(self.ctx.as_ptr()) }
    }

    /// n mmproj embd
    pub fn n_mmproj_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::clip_n_mmproj_embd(self.ctx.as_ptr()) }
    }

    /// image preprocess
    pub fn image_preprocess(
        &mut self,
        img: &ClipImageU8,
        res_imgs: &mut ClipImageF32Batch,
    ) -> bool {
        unsafe {
            llama_cpp_sys_2::clip_image_preprocess(
                self.ctx.as_ptr(),
                img.image.as_ptr(),
                res_imgs.as_mut_ptr(),
            )
        }
    }

    /// get newline tensor
    pub fn get_newline_tensor(&self) -> *mut llama_cpp_sys_2::ggml_tensor {
        unsafe {
            let newline_tensor = llama_cpp_sys_2::clip_get_newline_tensor(self.ctx.as_ptr());

            newline_tensor
        }
    }

    /// image encode
    pub fn image_encode(
        &mut self,
        n_threads: c_int,
        img: &mut ClipImageF32,
        vec: *mut f32,
    ) -> bool {
        unsafe {
            llama_cpp_sys_2::clip_image_encode(
                self.ctx.as_ptr(),
                n_threads,
                img.image.as_mut(),
                vec,
            )
        }
    }

    /// image batch encode
    pub fn image_batch_encode(
        &mut self,
        n_threads: c_int,
        img: &ClipImageF32Batch,
        vec: *mut f32,
    ) -> bool {
        unsafe {
            llama_cpp_sys_2::clip_image_batch_encode(
                self.ctx.as_ptr(),
                n_threads,
                &img.batch as *const llama_cpp_sys_2::clip_image_f32_batch,
                vec,
            )
        }
    }
}

impl Drop for ClipCtx {
    fn drop(&mut self) {
        unsafe {
            let ctx = self.ctx.as_ptr();
            llama_cpp_sys_2::clip_free(ctx);
        }
    }
}

/// A safe wrapper around `clip_image_u8`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipImageU8 {
    pub(crate) image: NonNull<llama_cpp_sys_2::clip_image_u8>,
}

impl ClipImageU8 {
    /// new
    pub fn new(_: &LlamaBackend) -> Result<Self, LlamaModelLoadError> {
        unsafe {
            let image = llama_cpp_sys_2::clip_image_u8_init();
            let image = NonNull::new(image).ok_or(LlamaModelLoadError::NullResult)?;
            tracing::debug!("Created image");
            Ok(Self { image })
        }
    }

    /// load image from a path
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(&mut self, path: impl AsRef<Path>) -> Result<bool, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;
        let cstr = CString::new(path)?;
        let loaded = unsafe {
            llama_cpp_sys_2::clip_image_load_from_file(cstr.as_ptr(), self.image.as_mut())
        };

        Ok(loaded)
    }

    /// load image from a path
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_bytes(&mut self, data: &[u8]) -> Result<bool, LlamaModelLoadError> {
        let loaded = unsafe {
            llama_cpp_sys_2::clip_image_load_from_bytes(
                data.as_ptr(),
                data.len(),
                self.image.as_mut(),
            )
        };

        Ok(loaded)
    }
}

impl Drop for ClipImageU8 {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::clip_image_u8_free(self.image.as_ptr()) }
    }
}

/// A safe wrapper around `clip_image_f32`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipImageF32 {
    pub(crate) image: NonNull<llama_cpp_sys_2::clip_image_f32>,
}

impl ClipImageF32 {
    /// new instance
    pub fn new(_: &LlamaBackend) -> Result<Self, LlamaModelLoadError> {
        unsafe {
            let image = llama_cpp_sys_2::clip_image_f32_init();
            let image = NonNull::new(image).ok_or(LlamaModelLoadError::NullResult)?;
            tracing::debug!("Created image batch");
            Ok(Self { image })
        }
    }
}

impl Drop for ClipImageF32 {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::clip_image_f32_free(self.image.as_ptr()) }
    }
}

/// A safe wrapper around `clip_image_u8_batch`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipImageU8Batch {
    pub(crate) batch: llama_cpp_sys_2::clip_image_u8_batch,
}

impl ClipImageU8Batch {
    /// new instance
    pub fn new() -> Self {
        let batch = llama_cpp_sys_2::clip_image_u8_batch {
            data: std::ptr::null_mut(),
            size: 0,
        };

        Self { batch }
    }

    fn as_mut_ptr(&mut self) -> *mut llama_cpp_sys_2::clip_image_u8_batch {
        let p = &self.batch as *const llama_cpp_sys_2::clip_image_u8_batch
            as *mut llama_cpp_sys_2::clip_image_u8_batch;
        p
    }
}

impl Drop for ClipImageU8Batch {
    fn drop(&mut self) {
        unsafe {
            let batch = self.as_mut_ptr();
            llama_cpp_sys_2::clip_image_u8_batch_free(batch)
        }
    }
}

/// A safe wrapper around `clip_image_u8_batch`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipImageF32Batch {
    pub(crate) batch: llama_cpp_sys_2::clip_image_f32_batch,
}

impl ClipImageF32Batch {
    /// new instance
    pub fn new() -> Self {
        let batch = llama_cpp_sys_2::clip_image_f32_batch {
            data: std::ptr::null_mut(),
            size: 0,
        };

        Self { batch }
    }

    fn as_mut_ptr(&mut self) -> *mut llama_cpp_sys_2::clip_image_f32_batch {
        let p = &self.batch as *const llama_cpp_sys_2::clip_image_f32_batch
            as *mut llama_cpp_sys_2::clip_image_f32_batch;
        p
    }
}

impl Drop for ClipImageF32Batch {
    fn drop(&mut self) {
        let batch = self.as_mut_ptr();
        unsafe { llama_cpp_sys_2::clip_image_f32_batch_free(batch) }
    }
}

/// clip_model_quantize
pub fn model_quantize(
    path_in: impl AsRef<Path>,
    path_out: impl AsRef<Path>,
    itype: c_int,
) -> Result<bool, LlamaModelLoadError> {
    let path_in = path_in.as_ref();
    let path_out = path_out.as_ref();
    let path_in = path_in
        .to_str()
        .ok_or(LlamaModelLoadError::PathToStrError(path_in.to_path_buf()))?;
    let path_out = path_out
        .to_str()
        .ok_or(LlamaModelLoadError::PathToStrError(path_out.to_path_buf()))?;

    let cstr_in = CString::new(path_in)?;
    let cstr_out = CString::new(path_out)?;
    unsafe {
        let ret = llama_cpp_sys_2::clip_model_quantize(cstr_in.as_ptr(), cstr_out.as_ptr(), itype);
        Ok(ret)
    }
}
