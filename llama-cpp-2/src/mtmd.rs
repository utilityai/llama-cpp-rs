use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::slice;

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// Input chunk types for multimodal data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdInputChunkType {
    Text = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT as isize,
    Image = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE as isize,
    Audio = llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO as isize,
}

impl From<llama_cpp_sys_2::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(chunk_type: llama_cpp_sys_2::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdInputChunkType::Text,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdInputChunkType::Image,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdInputChunkType::Audio,
            _ => panic!("Unknown MTMD input chunk type"),
        }
    }
}

/// Configuration parameters for MTMD context
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: i32,
    pub media_marker: CString,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        Self {
            use_gpu: false,
            print_timings: true,
            n_threads: 4,
            media_marker: CString::new(mtmd_default_marker()).unwrap_or_default(),
        }
    }
}

impl From<&MtmdContextParams> for llama_cpp_sys_2::mtmd_context_params {
    fn from(params: &MtmdContextParams) -> Self {
        let mut context = unsafe { llama_cpp_sys_2::mtmd_context_params_default() };

        context.use_gpu = params.use_gpu;
        context.print_timings = params.print_timings;
        context.n_threads = params.n_threads;
        context.media_marker = params.media_marker.as_ptr();

        context
    }
}

/// Text input configuration
#[derive(Debug, Clone)]
pub struct MtmdInputText {
    pub text: String,
    pub add_special: bool,
    pub parse_special: bool,
}

/// Safe wrapper around `mtmd_context`
pub struct MtmdContext {
    pub(crate) context: NonNull<llama_cpp_sys_2::mtmd_context>,
}

impl MtmdContext {
    /// Initialize MTMD context from a multimodal projection file
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = llama_cpp_sys_2::mtmd_context_params::from(&params);

        let context = unsafe {
            llama_cpp_sys_2::mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
            )
        };

        if context.is_null() {
            return Err(MtmdInitError::NullResult);
        }

        let context = NonNull::new(context).ok_or(MtmdInitError::NullResult)?;
        Ok(Self { context })
    }

    /// Check if non-causal mask is needed before llama_decode
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_decode_use_non_causal(self.context.as_ptr()) }
    }

    /// Check if the model uses M-RoPE for llama_decode
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check if the model supports vision input
    pub fn support_vision(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check if the model supports audio input
    pub fn support_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Tokenize input text and bitmaps into chunks
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new();
        let text_cstring = CString::new(text.text).unwrap_or_default();
        let input_text = llama_cpp_sys_2::mtmd_input_text {
            text: text_cstring.as_ptr(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        // Create bitmap pointers
        let bitmap_ptrs: Vec<*const llama_cpp_sys_2::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.bitmap.as_ptr() as *const _)
            .collect();

        let result = unsafe {
            llama_cpp_sys_2::mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &input_text,
                bitmap_ptrs.as_ptr() as *mut *const llama_cpp_sys_2::mtmd_bitmap,
                bitmaps.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MtmdTokenizeError::BitmapCountMismatch),
            2 => Err(MtmdTokenizeError::ImagePreprocessingError),
            _ => Err(MtmdTokenizeError::UnknownError(result)),
        }
    }

    /// Encode a chunk (for image/audio processing)
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk) -> Result<(), MtmdEncodeError> {
        let result = unsafe {
            llama_cpp_sys_2::mtmd_encode_chunk(self.context.as_ptr(), chunk.chunk.as_ptr())
        };

        if result == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(result))
        }
    }

    /// Get output embeddings from the last encode pass
    pub fn get_output_embeddings(&self) -> Option<&[f32]> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_get_output_embd(self.context.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            // Note: The size calculation would need context about the model and chunk
            // For now, returning None when we can't determine size safely
            None
        }
    }
}

unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_free(self.context.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_bitmap`
#[derive(Debug, Clone)]
pub struct MtmdBitmap {
    pub(crate) bitmap: NonNull<llama_cpp_sys_2::mtmd_bitmap>,
}

impl MtmdBitmap {
    /// Create a bitmap from image data (RGB format)
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        if data.len() != (nx * ny * 3) as usize {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { llama_cpp_sys_2::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data (PCM F32 format)
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap =
            unsafe { llama_cpp_sys_2::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file
    pub fn from_file(ctx: &MtmdContext, path: &str) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        let bitmap = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
            )
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data
    pub fn from_buffer(ctx: &MtmdContext, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe {
            llama_cpp_sys_2::mtmd_helper_bitmap_init_from_buf(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
            )
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Get bitmap width
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as bytes
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    /// Check if this is an audio bitmap
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Get bitmap ID (if set)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Set bitmap ID
    pub fn set_id(&self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_sys_2::mtmd_bitmap_set_id(self.bitmap.as_ptr(), id_cstr.as_ptr());
        }
        Ok(())
    }
}

unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunks`
pub struct MtmdInputChunks {
    pub(crate) chunks: NonNull<llama_cpp_sys_2::mtmd_input_chunks>,
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection
    pub fn new() -> Self {
        let chunks = unsafe { llama_cpp_sys_2::mtmd_input_chunks_init() };
        let chunks = NonNull::new(chunks).unwrap();
        Self { chunks }
    }

    /// Get the number of chunks
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if chunks collection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr =
            unsafe { llama_cpp_sys_2::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        if chunk_ptr.is_null() {
            None
        } else {
            // Note: We don't own this chunk, it's owned by the chunks collection
            Some(MtmdInputChunk {
                chunk: NonNull::new(chunk_ptr as *mut _).unwrap(),
                owned: false,
            })
        }
    }

    /// Get total number of tokens across all chunks
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// Get total position count across all chunks
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Evaluate chunks using the multimodal context and LLAMA context
    /// Returns the new n_past value on success
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        n_past: llama_cpp_sys_2::llama_pos,
        seq_id: llama_cpp_sys_2::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_sys_2::llama_pos, MtmdEvalError> {
        let mut new_n_past: llama_cpp_sys_2::llama_pos = 0;

        let result = unsafe {
            llama_cpp_sys_2::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &mut new_n_past,
            )
        };

        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunk`
pub struct MtmdInputChunk {
    pub(crate) chunk: NonNull<llama_cpp_sys_2::mtmd_input_chunk>,
    owned: bool,
}

impl MtmdInputChunk {
    /// Get the type of this chunk
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let chunk_type = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::from(chunk_type)
    }

    /// Get text tokens (only valid for text chunks)
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_sys_2::mtmd_input_chunk_get_tokens_text(self.chunk.as_ptr(), &mut n_tokens)
        };

        if tokens_ptr.is_null() || n_tokens == 0 {
            None
        } else {
            unsafe {
                Some(slice::from_raw_parts(
                    tokens_ptr as *const LlamaToken,
                    n_tokens,
                ))
            }
        }
    }

    /// Get the number of tokens in this chunk
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID (if available)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Create a copy of this chunk that you own
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_sys_2::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::NullResult)?;
        Ok(Self { chunk, owned: true })
    }
}

impl Drop for MtmdInputChunk {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_sys_2::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

/// Get the default media marker
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = llama_cpp_sys_2::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

// Error types
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("MTMD context initialization returned null")]
    NullResult,
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    #[error("Failed to create CString: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    #[error("Bitmap creation returned null")]
    NullResult,
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    #[error("Input chunks creation returned null")]
    NullResult,
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    #[error("Input chunk operation returned null")]
    NullResult,
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}

#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}
