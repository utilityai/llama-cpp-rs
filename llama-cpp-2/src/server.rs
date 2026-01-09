//! Server module providing safe wrappers around the llama-server infrastructure.
//!
//! This module allows direct in-process integration with the llama-server, similar to
//! how `cli.cpp` works in the upstream llama.cpp project, without using HTTP.
//!
//! # Example
//!
//! ```ignore
//! use llama_cpp_2::server::{ServerContext, ServerTaskParams};
//! use std::thread;
//!
//! // Create and configure server context
//! let mut ctx = ServerContext::new();
//! ctx.load_model("path/to/model.gguf", Default::default())?;
//!
//! // Start the server loop in a background thread
//! let ctx_clone = ctx.clone();
//! let handle = thread::spawn(move || {
//!     ctx_clone.start_loop();
//! });
//!
//! // Get a response reader and submit tasks
//! let mut reader = ctx.get_response_reader()?;
//! let task_id = reader.get_new_id();
//! 
//! let params = ServerTaskParams::default();
//! let messages = r#"[{"role": "user", "content": "Hello!"}]"#;
//! reader.post_completion(task_id, &params, messages, &[])?;
//!
//! // Read streaming results
//! while let Some(result) = reader.next(|| false)? {
//!     if let Some(diffs) = result.get_diffs() {
//!         for diff in diffs {
//!             if let Some(content) = diff.content_delta {
//!                 print!("{}", content);
//!             }
//!         }
//!     }
//!     if result.is_stop() {
//!         break;
//!     }
//! }
//!
//! // Cleanup
//! ctx.terminate();
//! handle.join().unwrap();
//! ```

use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::Arc;
use thiserror::Error;

use llama_cpp_sys_2::{
    llama_server_context,
    llama_server_context_free,
    llama_server_context_get_llama_context,
    llama_server_context_get_meta,
    llama_server_context_load_model,
    llama_server_context_meta_free,
    llama_server_context_new,
    llama_server_context_start_loop,
    llama_server_context_terminate,
    llama_server_msg_diff,
    llama_server_response_reader,
    llama_server_response_reader_free,
    llama_server_response_reader_get_new_id,
    llama_server_response_reader_has_next,
    llama_server_response_reader_new,
    llama_server_response_reader_next,
    llama_server_response_reader_post_completion,
    llama_server_response_reader_stop,
    llama_server_result_timings,
    llama_server_string_free,
    llama_server_task_params,
    llama_server_task_params_default,
    llama_server_task_result,
    llama_server_task_result_free,
    llama_server_task_result_get_content,
    llama_server_task_result_get_diff,
    llama_server_task_result_get_diff_count,
    llama_server_task_result_get_error,
    llama_server_task_result_get_timings,
    llama_server_task_result_get_type,
    llama_server_task_result_is_error,
    llama_server_task_result_is_stop,
    llama_server_task_result_to_json,
    LLAMA_SERVER_RESULT_TYPE_ERROR,
    LLAMA_SERVER_RESULT_TYPE_FINAL,
    LLAMA_SERVER_RESULT_TYPE_PARTIAL,
};

/// Errors that can occur when using the server infrastructure.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Failed to create server context
    #[error("Failed to create server context")]
    ContextCreationFailed,

    /// Failed to load model
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    /// Failed to create response reader
    #[error("Failed to create response reader")]
    ReaderCreationFailed,

    /// Failed to post task
    #[error("Failed to post task: {0}")]
    PostTaskFailed(String),

    /// Invalid JSON
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    /// Task error
    #[error("Task error: {0}")]
    TaskError(String),

    /// Null pointer received
    #[error("Null pointer received")]
    NullPointer,

    /// String conversion error
    #[error("String conversion error: {0}")]
    StringConversion(String),
}

/// Result type for server operations.
pub type ServerResult<T> = std::result::Result<T, ServerError>;

/// Flash attention mode.
/// 
/// Controls whether flash attention is used for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum FlashAttnType {
    /// Automatically determine whether to use flash attention (default)
    #[default]
    Auto = -1,
    /// Disable flash attention
    Disabled = 0,
    /// Enable flash attention
    Enabled = 1,
}

/// Model loading parameters for the server context.
#[derive(Debug, Clone)]
pub struct ServerModelParams {
    /// Context size (0 = use model default)
    pub n_ctx: i32,
    /// Batch size for prompt processing
    pub n_batch: i32,
    /// Micro-batch size
    pub n_ubatch: i32,
    /// Number of threads for generation
    pub n_threads: i32,
    /// Number of threads for batch processing
    pub n_threads_batch: i32,
    /// Number of layers to offload to GPU
    pub n_gpu_layers: i32,
    /// Use memory mapping
    pub use_mmap: bool,
    /// Lock model in memory
    pub use_mlock: bool,
    /// Flash attention type: Auto, Disabled, or Enabled
    pub flash_attn_type: FlashAttnType,
    /// Custom chat template (None = use model default)
    pub chat_template: Option<String>,
    /// System prompt
    pub system_prompt: Option<String>,
}

impl Default for ServerModelParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: -1, // auto-detect
            n_threads_batch: -1,
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
            flash_attn_type: FlashAttnType::Auto,
            chat_template: None,
            system_prompt: None,
        }
    }
}

/// Task parameters for completion requests.
#[derive(Debug, Clone)]
pub struct ServerTaskParams {
    /// Enable streaming mode
    pub stream: bool,
    /// Cache the prompt for reuse
    pub cache_prompt: bool,
    /// Return generated tokens
    pub return_tokens: bool,
    /// Return progress updates
    pub return_progress: bool,
    /// Include per-token timing information
    pub timings_per_token: bool,
    /// Include post-sampling probabilities
    pub post_sampling_probs: bool,
    /// Number of tokens to keep from initial prompt
    pub n_keep: i32,
    /// Number of tokens to discard when shifting context
    pub n_discard: i32,
    /// Maximum tokens to predict (-1 = unlimited)
    pub n_predict: i32,
    /// Minimum line indentation
    pub n_indent: i32,
    /// Random seed
    pub seed: i32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Min-p sampling
    pub min_p: f32,
    /// Typical-p sampling
    pub typical_p: f32,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Top-k sampling
    pub top_k: i32,
    /// Number of tokens to consider for repetition penalty
    pub repeat_last_n: i32,
    /// Mirostat mode (0 = disabled)
    pub mirostat: i32,
    /// Mirostat tau parameter
    pub mirostat_tau: f32,
    /// Mirostat eta parameter
    pub mirostat_eta: f32,
    /// Stop sequences
    pub antiprompt: Vec<String>,
}

impl Default for ServerTaskParams {
    fn default() -> Self {
        Self {
            stream: true,
            cache_prompt: true,
            return_tokens: false,
            return_progress: false,
            timings_per_token: true,
            post_sampling_probs: false,
            n_keep: 0,
            n_discard: 0,
            n_predict: -1,
            n_indent: 0,
            seed: u32::MAX as i32,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            top_k: 40,
            repeat_last_n: 64,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            antiprompt: Vec::new(),
        }
    }
}

impl ServerTaskParams {
    /// Convert to FFI struct, returning the struct and owned CStrings that must live as long as the struct is used.
    fn to_ffi(&self) -> (llama_server_task_params, Vec<CString>, Vec<*const i8>) {
        let mut params = unsafe { llama_server_task_params_default() };

        params.stream = self.stream;
        params.cache_prompt = self.cache_prompt;
        params.return_tokens = self.return_tokens;
        params.return_progress = self.return_progress;
        params.timings_per_token = self.timings_per_token;
        params.post_sampling_probs = self.post_sampling_probs;
        params.n_keep = self.n_keep;
        params.n_discard = self.n_discard;
        params.n_predict = self.n_predict;
        params.n_indent = self.n_indent;
        params.seed = self.seed;
        params.temperature = self.temperature;
        params.top_p = self.top_p;
        params.min_p = self.min_p;
        params.typical_p = self.typical_p;
        params.repeat_penalty = self.repeat_penalty;
        params.presence_penalty = self.presence_penalty;
        params.frequency_penalty = self.frequency_penalty;
        params.top_k = self.top_k;
        params.repeat_last_n = self.repeat_last_n;
        params.mirostat = self.mirostat;
        params.mirostat_tau = self.mirostat_tau;
        params.mirostat_eta = self.mirostat_eta;

        // Convert antiprompt strings
        let cstrings: Vec<CString> = self
            .antiprompt
            .iter()
            .filter_map(|s| CString::new(s.as_str()).ok())
            .collect();

        let ptrs: Vec<*const i8> = cstrings.iter().map(|cs| cs.as_ptr()).collect();

        if !ptrs.is_empty() {
            // Cast to mutable pointer - the C code doesn't actually mutate this array
            params.antiprompt = ptrs.as_ptr() as *mut *const i8;
            params.antiprompt_count = ptrs.len();
        }

        (params, cstrings, ptrs)
    }
}

/// Metadata about the loaded model and server.
#[derive(Debug, Clone)]
pub struct ServerContextMeta {
    /// Build information string
    pub build_info: String,
    /// Model name
    pub model_name: String,
    /// Path to the model file
    pub model_path: String,
    /// Whether multimodal support is available
    pub has_mtmd: bool,
    /// Whether image input is supported
    pub has_inp_image: bool,
    /// Whether audio input is supported
    pub has_inp_audio: bool,
    /// Context size per slot
    pub slot_n_ctx: i32,
    /// Pooling type
    pub pooling_type: i32,
    /// Chat template
    pub chat_template: String,
    /// Beginning of sequence token string
    pub bos_token_str: String,
    /// End of sequence token string
    pub eos_token_str: String,
    /// Number of tokens in vocabulary
    pub model_vocab_n_tokens: i32,
    /// Training context size
    pub model_n_ctx_train: i32,
    /// Embedding input dimension
    pub model_n_embd_inp: i32,
    /// Number of model parameters
    pub model_n_params: u64,
    /// Model size in bytes
    pub model_size: u64,
}

/// Timing information for a completion.
#[derive(Debug, Clone, Default)]
pub struct ResultTimings {
    /// Number of cached tokens
    pub cache_n: i32,
    /// Number of prompt tokens
    pub prompt_n: i32,
    /// Prompt processing time in milliseconds
    pub prompt_ms: f64,
    /// Time per prompt token in milliseconds
    pub prompt_per_token_ms: f64,
    /// Prompt tokens per second
    pub prompt_per_second: f64,
    /// Number of predicted tokens
    pub predicted_n: i32,
    /// Prediction time in milliseconds
    pub predicted_ms: f64,
    /// Time per predicted token in milliseconds
    pub predicted_per_token_ms: f64,
    /// Predicted tokens per second
    pub predicted_per_second: f64,
    /// Number of draft tokens (speculative)
    pub draft_n: i32,
    /// Number of accepted draft tokens
    pub draft_n_accepted: i32,
}

impl From<llama_server_result_timings> for ResultTimings {
    fn from(t: llama_server_result_timings) -> Self {
        Self {
            cache_n: t.cache_n,
            prompt_n: t.prompt_n,
            prompt_ms: t.prompt_ms,
            prompt_per_token_ms: t.prompt_per_token_ms,
            prompt_per_second: t.prompt_per_second,
            predicted_n: t.predicted_n,
            predicted_ms: t.predicted_ms,
            predicted_per_token_ms: t.predicted_per_token_ms,
            predicted_per_second: t.predicted_per_second,
            draft_n: t.draft_n,
            draft_n_accepted: t.draft_n_accepted,
        }
    }
}

/// A single message diff from streaming results.
#[derive(Debug, Clone, Default)]
pub struct MessageDiff {
    /// New content added
    pub content_delta: Option<String>,
    /// New reasoning content added
    pub reasoning_content_delta: Option<String>,
    /// Tool call ID
    pub tool_call_id: Option<String>,
    /// Tool call function name
    pub tool_call_name: Option<String>,
    /// Tool call arguments
    pub tool_call_arguments: Option<String>,
}

/// Result type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskResultType {
    /// Unknown result type
    Unknown,
    /// Error result
    Error,
    /// Partial (streaming) result
    Partial,
    /// Final result
    Final,
}

/// A task result from the server.
pub struct TaskResult {
    ptr: NonNull<llama_server_task_result>,
    result_type: TaskResultType,
}

// Safety: TaskResult only contains a pointer that is exclusively owned
unsafe impl Send for TaskResult {}

impl TaskResult {
    /// Get the result type.
    pub fn result_type(&self) -> TaskResultType {
        self.result_type
    }

    /// Check if this is an error result.
    pub fn is_error(&self) -> bool {
        unsafe { llama_server_task_result_is_error(self.ptr.as_ptr()) }
    }

    /// Check if this is a final (stop) result.
    pub fn is_stop(&self) -> bool {
        unsafe { llama_server_task_result_is_stop(self.ptr.as_ptr()) }
    }

    /// Get the error message if this is an error result.
    pub fn get_error(&self) -> Option<String> {
        unsafe {
            let ptr = llama_server_task_result_get_error(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                llama_server_string_free(ptr);
                Some(s)
            }
        }
    }

    /// Get timing information.
    pub fn get_timings(&self) -> ResultTimings {
        unsafe { llama_server_task_result_get_timings(self.ptr.as_ptr()).into() }
    }

    /// Get message diffs (for streaming partial results).
    pub fn get_diffs(&self) -> Vec<MessageDiff> {
        unsafe {
            let count = llama_server_task_result_get_diff_count(self.ptr.as_ptr());
            let mut diffs = Vec::with_capacity(count);

            for i in 0..count {
                let mut diff = llama_server_msg_diff {
                    content_delta: std::ptr::null(),
                    reasoning_content_delta: std::ptr::null(),
                    tool_call_id: std::ptr::null(),
                    tool_call_name: std::ptr::null(),
                    tool_call_arguments: std::ptr::null(),
                };

                if llama_server_task_result_get_diff(self.ptr.as_ptr(), i, &mut diff) {
                    let msg_diff = MessageDiff {
                        content_delta: ptr_to_option_string(diff.content_delta),
                        reasoning_content_delta: ptr_to_option_string(diff.reasoning_content_delta),
                        tool_call_id: ptr_to_option_string(diff.tool_call_id),
                        tool_call_name: ptr_to_option_string(diff.tool_call_name),
                        tool_call_arguments: ptr_to_option_string(diff.tool_call_arguments),
                    };
                    diffs.push(msg_diff);
                }
            }

            diffs
        }
    }

    /// Get the full generated content (for final results).
    pub fn get_content(&self) -> Option<String> {
        unsafe {
            let ptr = llama_server_task_result_get_content(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                llama_server_string_free(ptr);
                Some(s)
            }
        }
    }

    /// Get the result as a JSON string.
    pub fn to_json(&self) -> Option<String> {
        unsafe {
            let ptr = llama_server_task_result_to_json(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                llama_server_string_free(ptr);
                Some(s)
            }
        }
    }
}

impl Drop for TaskResult {
    fn drop(&mut self) {
        unsafe {
            llama_server_task_result_free(self.ptr.as_ptr());
        }
    }
}

/// Helper to convert C string pointer to Option<String>
fn ptr_to_option_string(ptr: *const i8) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(CStr::from_ptr(ptr).to_string_lossy().into_owned()) }
    }
}

/// A response reader for streaming task results.
pub struct ResponseReader {
    ptr: NonNull<llama_server_response_reader>,
}

// Safety: ResponseReader owns its pointer exclusively
unsafe impl Send for ResponseReader {}

impl ResponseReader {
    /// Get a new unique task ID.
    pub fn get_new_id(&mut self) -> i32 {
        unsafe { llama_server_response_reader_get_new_id(self.ptr.as_ptr()) }
    }

    /// Check if there are more results to read.
    pub fn has_next(&self) -> bool {
        unsafe { llama_server_response_reader_has_next(self.ptr.as_ptr()) }
    }

    /// Post a completion task.
    ///
    /// # Arguments
    /// * `task_id` - Unique task ID from `get_new_id()`
    /// * `params` - Task parameters
    /// * `messages_json` - JSON array of chat messages, e.g. `[{"role": "user", "content": "Hello"}]`
    /// * `files` - Optional file buffers for multimodal input
    pub fn post_completion(
        &mut self,
        task_id: i32,
        params: &ServerTaskParams,
        messages_json: &str,
        files: &[&[u8]],
    ) -> ServerResult<()> {
        let messages_cstr = CString::new(messages_json)
            .map_err(|e| ServerError::InvalidJson(e.to_string()))?;

        let (ffi_params, _cstrings, _ptrs) = params.to_ffi();

        // Prepare file buffers
        let file_ptrs: Vec<*const u8> = files.iter().map(|f| f.as_ptr()).collect();
        let file_sizes: Vec<usize> = files.iter().map(|f| f.len()).collect();

        let success = unsafe {
            llama_server_response_reader_post_completion(
                self.ptr.as_ptr(),
                task_id,
                &ffi_params,
                messages_cstr.as_ptr(),
                if files.is_empty() {
                    std::ptr::null_mut()
                } else {
                    // Cast to mutable pointer - the C code doesn't actually mutate this array
                    file_ptrs.as_ptr() as *mut *const u8
                },
                if files.is_empty() {
                    std::ptr::null()
                } else {
                    file_sizes.as_ptr()
                },
                files.len(),
            )
        };

        if success {
            Ok(())
        } else {
            Err(ServerError::PostTaskFailed(
                "Failed to post completion task".to_string(),
            ))
        }
    }

    /// Get the next result, blocking until available.
    ///
    /// # Arguments
    /// * `should_stop` - Callback that returns true to cancel waiting
    ///
    /// # Returns
    /// * `Ok(Some(result))` - A result is available
    /// * `Ok(None)` - No more results or stopped
    /// * `Err(_)` - An error occurred
    pub fn next<F>(&mut self, should_stop: F) -> ServerResult<Option<TaskResult>>
    where
        F: Fn() -> bool + 'static,
    {
        // We need to pass the closure to C, so we box it
        let boxed: Box<dyn Fn() -> bool> = Box::new(should_stop);
        let user_data = Box::into_raw(Box::new(boxed)) as *mut std::ffi::c_void;

        extern "C" fn stop_callback(user_data: *mut std::ffi::c_void) -> bool {
            unsafe {
                let closure = &*(user_data as *const Box<dyn Fn() -> bool>);
                closure()
            }
        }

        let result_ptr = unsafe {
            llama_server_response_reader_next(self.ptr.as_ptr(), Some(stop_callback), user_data)
        };

        // Clean up the boxed closure
        unsafe {
            drop(Box::from_raw(user_data as *mut Box<dyn Fn() -> bool>));
        }

        if result_ptr.is_null() {
            return Ok(None);
        }

        let ptr = NonNull::new(result_ptr).ok_or(ServerError::NullPointer)?;

        let result_type = unsafe {
            match llama_server_task_result_get_type(result_ptr) {
                t if t == LLAMA_SERVER_RESULT_TYPE_ERROR => {
                    TaskResultType::Error
                }
                t if t == LLAMA_SERVER_RESULT_TYPE_PARTIAL => {
                    TaskResultType::Partial
                }
                t if t == LLAMA_SERVER_RESULT_TYPE_FINAL => {
                    TaskResultType::Final
                }
                _ => TaskResultType::Unknown,
            }
        };

        Ok(Some(TaskResult { ptr, result_type }))
    }

    /// Stop/cancel the reader.
    pub fn stop(&mut self) {
        unsafe {
            llama_server_response_reader_stop(self.ptr.as_ptr());
        }
    }
}

impl Drop for ResponseReader {
    fn drop(&mut self) {
        unsafe {
            llama_server_response_reader_free(self.ptr.as_ptr());
        }
    }
}

/// Inner server context data (shared via Arc).
struct ServerContextInner {
    ptr: NonNull<llama_server_context>,
}

// Safety: The server context is thread-safe for the operations we expose
unsafe impl Send for ServerContextInner {}
unsafe impl Sync for ServerContextInner {}

impl Drop for ServerContextInner {
    fn drop(&mut self) {
        unsafe {
            llama_server_context_free(self.ptr.as_ptr());
        }
    }
}

/// The main server context for managing inference.
///
/// This is a thread-safe handle that can be cloned and shared between threads.
/// The actual context is reference-counted and will be freed when the last
/// handle is dropped.
#[derive(Clone)]
pub struct ServerContext {
    inner: Arc<ServerContextInner>,
}

impl ServerContext {
    /// Create a new server context.
    pub fn new() -> ServerResult<Self> {
        let ptr = unsafe { llama_server_context_new() };
        let ptr = NonNull::new(ptr).ok_or(ServerError::ContextCreationFailed)?;

        Ok(Self {
            inner: Arc::new(ServerContextInner { ptr }),
        })
    }

    /// Load a model into the server context.
    pub fn load_model(&self, model_path: &str, params: ServerModelParams) -> ServerResult<()> {
        let model_path_cstr = CString::new(model_path)
            .map_err(|e| ServerError::ModelLoadFailed(e.to_string()))?;

        let chat_template_cstr = params
            .chat_template
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()
            .map_err(|e| ServerError::ModelLoadFailed(e.to_string()))?;

        let system_prompt_cstr = params
            .system_prompt
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()
            .map_err(|e| ServerError::ModelLoadFailed(e.to_string()))?;

        let success = unsafe {
            llama_server_context_load_model(
                self.inner.ptr.as_ptr(),
                model_path_cstr.as_ptr(),
                params.n_ctx,
                params.n_batch,
                params.n_ubatch,
                params.n_threads,
                params.n_threads_batch,
                params.n_gpu_layers,
                params.use_mmap,
                params.use_mlock,
                params.flash_attn_type as i32,
                chat_template_cstr
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
                system_prompt_cstr
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };

        if success {
            Ok(())
        } else {
            Err(ServerError::ModelLoadFailed(format!(
                "Failed to load model: {}",
                model_path
            )))
        }
    }

    /// Start the server loop. This is blocking and should be run in a separate thread.
    pub fn start_loop(&self) {
        unsafe {
            llama_server_context_start_loop(self.inner.ptr.as_ptr());
        }
    }

    /// Terminate the server loop.
    pub fn terminate(&self) {
        unsafe {
            llama_server_context_terminate(self.inner.ptr.as_ptr());
        }
    }

    /// Get the underlying llama_context pointer (may be null if sleeping).
    pub fn get_llama_context(&self) -> *mut llama_cpp_sys_2::llama_context {
        unsafe { llama_server_context_get_llama_context(self.inner.ptr.as_ptr()) }
    }

    /// Get metadata about the loaded model and server.
    pub fn get_meta(&self) -> ServerResult<ServerContextMeta> {
        let meta_ptr = unsafe { llama_server_context_get_meta(self.inner.ptr.as_ptr()) };

        if meta_ptr.is_null() {
            return Err(ServerError::NullPointer);
        }

        let meta = unsafe {
            let m = &*meta_ptr;
            ServerContextMeta {
                build_info: ptr_to_option_string(m.build_info).unwrap_or_default(),
                model_name: ptr_to_option_string(m.model_name).unwrap_or_default(),
                model_path: ptr_to_option_string(m.model_path).unwrap_or_default(),
                has_mtmd: m.has_mtmd,
                has_inp_image: m.has_inp_image,
                has_inp_audio: m.has_inp_audio,
                slot_n_ctx: m.slot_n_ctx,
                pooling_type: m.pooling_type,
                chat_template: ptr_to_option_string(m.chat_template).unwrap_or_default(),
                bos_token_str: ptr_to_option_string(m.bos_token_str).unwrap_or_default(),
                eos_token_str: ptr_to_option_string(m.eos_token_str).unwrap_or_default(),
                model_vocab_n_tokens: m.model_vocab_n_tokens,
                model_n_ctx_train: m.model_n_ctx_train,
                model_n_embd_inp: m.model_n_embd_inp,
                model_n_params: m.model_n_params,
                model_size: m.model_size,
            }
        };

        unsafe {
            llama_server_context_meta_free(meta_ptr);
        }

        Ok(meta)
    }

    /// Get a new response reader for submitting tasks and reading results.
    pub fn get_response_reader(&self) -> ServerResult<ResponseReader> {
        let ptr = unsafe { llama_server_response_reader_new(self.inner.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(ServerError::ReaderCreationFailed)?;

        Ok(ResponseReader { ptr })
    }
}
