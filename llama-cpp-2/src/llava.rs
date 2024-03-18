//! safe wrapper for multimodel model `llava`.

mod clip;

use std::{
    ffi::{c_int, CString},
    path::Path,
    ptr::NonNull,
};

pub use clip::*;

use crate::{context::LlamaContext, LlamaModelLoadError};

/// A safe wrapper around `llava_image_embed`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlavaImageEmbed {
    pub(crate) embed: NonNull<llama_cpp_sys_2::llava_image_embed>,
}

impl LlavaImageEmbed {
    /// make with bytes
    pub fn make_with_bytes(
        ctx_clip: &mut ClipCtx,
        n_threads: c_int,
        buff: &[u8],
    ) -> Result<Self, LlamaModelLoadError> {
        let embed = unsafe {
            llama_cpp_sys_2::llava_image_embed_make_with_bytes(
                ctx_clip.ctx.as_mut(),
                n_threads,
                buff.as_ptr(),
                buff.len() as i32,
            )
        };
        let embed = NonNull::new(embed).ok_or(LlamaModelLoadError::NullResult)?;
        Ok(Self { embed })
    }

    /// make with filename
    pub fn make_with_file(
        ctx_clip: &mut ClipCtx,
        n_threads: c_int,
        path: impl AsRef<Path>,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let embed = unsafe {
            llama_cpp_sys_2::llava_image_embed_make_with_filename(
                ctx_clip.ctx.as_mut(),
                n_threads,
                cstr.as_ptr(),
            )
        };
        let embed = NonNull::new(embed).ok_or(LlamaModelLoadError::NullResult)?;
        Ok(Self { embed })
    }

    /// eval embed image
    pub fn eval(&self, llama_ctx: &mut LlamaContext, n_batch: c_int, n_past: &mut c_int) -> bool {
        unsafe {
            llama_cpp_sys_2::llava_eval_image_embed(
                llama_ctx.context.as_mut(),
                self.embed.as_ptr(),
                n_batch,
                n_past,
            )
        }
    }
}

impl Drop for LlavaImageEmbed {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llava_image_embed_free(self.embed.as_ptr());
        }
    }
}
