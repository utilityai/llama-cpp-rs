//! A safe wrapper around `llama_context_params`.
use std::fmt::Debug;
use std::num::NonZeroU32;

/// A rusty wrapper around `rope_scaling_type`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    /// The scaling type is unspecified
    Unspecified = -1,
    /// No scaling
    None = 0,
    /// Linear scaling
    Linear = 1,
    /// Yarn scaling
    Yarn = 2,
}

/// Create a `RopeScalingType` from a `c_int` - returns `RopeScalingType::ScalingUnspecified` if
/// the value is not recognized.
impl From<i32> for RopeScalingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Linear,
            2 => Self::Yarn,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `RopeScalingType`.
impl From<RopeScalingType> for i32 {
    fn from(value: RopeScalingType) -> Self {
        match value {
            RopeScalingType::None => 0,
            RopeScalingType::Linear => 1,
            RopeScalingType::Yarn => 2,
            RopeScalingType::Unspecified => -1,
        }
    }
}

/// A rusty wrapper around `LLAMA_POOLING_TYPE`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaPoolingType {
    /// The pooling type is unspecified
    Unspecified = -1,
    /// No pooling
    None = 0,
    /// Mean pooling
    Mean = 1,
    /// CLS pooling
    Cls = 2,
    /// Last pooling
    Last = 3,
    /// Rank pooling
    Rank = 4,
}

/// Create a `LlamaPoolingType` from a `c_int` - returns `LlamaPoolingType::Unspecified` if
/// the value is not recognized.
impl From<i32> for LlamaPoolingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Mean,
            2 => Self::Cls,
            3 => Self::Last,
            4 => Self::Rank,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `LlamaPoolingType`.
impl From<LlamaPoolingType> for i32 {
    fn from(value: LlamaPoolingType) -> Self {
        match value {
            LlamaPoolingType::None => 0,
            LlamaPoolingType::Mean => 1,
            LlamaPoolingType::Cls => 2,
            LlamaPoolingType::Last => 3,
            LlamaPoolingType::Rank => 4,
            LlamaPoolingType::Unspecified => -1,
        }
    }
}

/// A rusty wrapper around `LLAMA_ATTENTION_TYPE`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaAttentionType {
    /// The attention type is unspecified
    Unspecified = -1,
    /// Causal attention
    Causal = 0,
    /// Non-causal attention
    NonCausal = 1,
}

impl From<i32> for LlamaAttentionType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Causal,
            1 => Self::NonCausal,
            _ => Self::Unspecified,
        }
    }
}

impl From<LlamaAttentionType> for i32 {
    fn from(value: LlamaAttentionType) -> Self {
        match value {
            LlamaAttentionType::Causal => 0,
            LlamaAttentionType::NonCausal => 1,
            LlamaAttentionType::Unspecified => -1,
        }
    }
}

/// A rusty wrapper around `ggml_type` for KV cache types.
#[expect(
    non_camel_case_types,
    reason = "variant names mirror llama.cpp's `enum ggml_type` symbol names verbatim so they can \
              be matched 1:1 against the C ABI without a translation table"
)]
#[expect(
    missing_docs,
    reason = "each variant denotes a quantisation flavour whose semantics are defined upstream in \
              ggml; restating the upstream spec inline would risk drifting from the source of truth"
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvCacheType {
    /// Represents an unknown or not-yet-mapped `ggml_type` and carries the raw value.
    /// When passed through FFI, the raw value is used as-is (if llama.cpp supports it,
    /// the runtime will operate with that type).
    /// This variant preserves API compatibility when new `ggml_type` values are
    /// introduced in the future.
    Unknown(llama_cpp_bindings_sys::ggml_type),
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
}

impl From<KvCacheType> for llama_cpp_bindings_sys::ggml_type {
    fn from(value: KvCacheType) -> Self {
        match value {
            KvCacheType::Unknown(raw) => raw,
            KvCacheType::F32 => llama_cpp_bindings_sys::GGML_TYPE_F32,
            KvCacheType::F16 => llama_cpp_bindings_sys::GGML_TYPE_F16,
            KvCacheType::Q4_0 => llama_cpp_bindings_sys::GGML_TYPE_Q4_0,
            KvCacheType::Q4_1 => llama_cpp_bindings_sys::GGML_TYPE_Q4_1,
            KvCacheType::Q5_0 => llama_cpp_bindings_sys::GGML_TYPE_Q5_0,
            KvCacheType::Q5_1 => llama_cpp_bindings_sys::GGML_TYPE_Q5_1,
            KvCacheType::Q8_0 => llama_cpp_bindings_sys::GGML_TYPE_Q8_0,
            KvCacheType::Q8_1 => llama_cpp_bindings_sys::GGML_TYPE_Q8_1,
            KvCacheType::Q2_K => llama_cpp_bindings_sys::GGML_TYPE_Q2_K,
            KvCacheType::Q3_K => llama_cpp_bindings_sys::GGML_TYPE_Q3_K,
            KvCacheType::Q4_K => llama_cpp_bindings_sys::GGML_TYPE_Q4_K,
            KvCacheType::Q5_K => llama_cpp_bindings_sys::GGML_TYPE_Q5_K,
            KvCacheType::Q6_K => llama_cpp_bindings_sys::GGML_TYPE_Q6_K,
            KvCacheType::Q8_K => llama_cpp_bindings_sys::GGML_TYPE_Q8_K,
            KvCacheType::IQ2_XXS => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS,
            KvCacheType::IQ2_XS => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS,
            KvCacheType::IQ3_XXS => llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS,
            KvCacheType::IQ1_S => llama_cpp_bindings_sys::GGML_TYPE_IQ1_S,
            KvCacheType::IQ4_NL => llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL,
            KvCacheType::IQ3_S => llama_cpp_bindings_sys::GGML_TYPE_IQ3_S,
            KvCacheType::IQ2_S => llama_cpp_bindings_sys::GGML_TYPE_IQ2_S,
            KvCacheType::IQ4_XS => llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS,
            KvCacheType::I8 => llama_cpp_bindings_sys::GGML_TYPE_I8,
            KvCacheType::I16 => llama_cpp_bindings_sys::GGML_TYPE_I16,
            KvCacheType::I32 => llama_cpp_bindings_sys::GGML_TYPE_I32,
            KvCacheType::I64 => llama_cpp_bindings_sys::GGML_TYPE_I64,
            KvCacheType::F64 => llama_cpp_bindings_sys::GGML_TYPE_F64,
            KvCacheType::IQ1_M => llama_cpp_bindings_sys::GGML_TYPE_IQ1_M,
            KvCacheType::BF16 => llama_cpp_bindings_sys::GGML_TYPE_BF16,
            KvCacheType::TQ1_0 => llama_cpp_bindings_sys::GGML_TYPE_TQ1_0,
            KvCacheType::TQ2_0 => llama_cpp_bindings_sys::GGML_TYPE_TQ2_0,
            KvCacheType::MXFP4 => llama_cpp_bindings_sys::GGML_TYPE_MXFP4,
        }
    }
}

impl From<llama_cpp_bindings_sys::ggml_type> for KvCacheType {
    fn from(value: llama_cpp_bindings_sys::ggml_type) -> Self {
        match value {
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F32 => Self::F32,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F16 => Self::F16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_0 => Self::Q4_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_1 => Self::Q4_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_0 => Self::Q5_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_1 => Self::Q5_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_0 => Self::Q8_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_1 => Self::Q8_1,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q2_K => Self::Q2_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q3_K => Self::Q3_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q4_K => Self::Q4_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q5_K => Self::Q5_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q6_K => Self::Q6_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_Q8_K => Self::Q8_K,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS => Self::IQ2_XXS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS => Self::IQ2_XS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS => Self::IQ3_XXS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ1_S => Self::IQ1_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL => Self::IQ4_NL,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ3_S => Self::IQ3_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ2_S => Self::IQ2_S,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS => Self::IQ4_XS,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I8 => Self::I8,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I16 => Self::I16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I32 => Self::I32,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_I64 => Self::I64,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_F64 => Self::F64,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_IQ1_M => Self::IQ1_M,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_BF16 => Self::BF16,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_TQ1_0 => Self::TQ1_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_TQ2_0 => Self::TQ2_0,
            x if x == llama_cpp_bindings_sys::GGML_TYPE_MXFP4 => Self::MXFP4,
            _ => Self::Unknown(value),
        }
    }
}

/// A safe wrapper around `llama_context_params`.
///
/// Generally this should be created with [`Default::default()`] and then modified with `with_*` methods.
///
/// # Examples
///
/// ```rust
/// # use std::num::NonZeroU32;
/// use llama_cpp_bindings::context::params::LlamaContextParams;
///
///let ctx_params = LlamaContextParams::default()
///    .with_n_ctx(NonZeroU32::new(2048));
///
/// assert_eq!(ctx_params.n_ctx(), NonZeroU32::new(2048));
/// ```
#[derive(Debug, Clone)]
#[expect(
    missing_docs,
    reason = "field meanings mirror llama.cpp's `llama_context_params` C struct; restating each \
              one inline would risk drift from the upstream spec — the doc-comment on the struct \
              points at the canonical reference"
)]
#[expect(
    clippy::module_name_repetitions,
    reason = "`LlamaContextParams` is the canonical Rust name in the public API; renaming it to \
              `Params` would force `params::Params` at every call site"
)]
pub struct LlamaContextParams {
    pub context_params: llama_cpp_bindings_sys::llama_context_params,
}

/// SAFETY: we do not currently allow setting or reading the pointers that cause this to not be automatically send or sync.
unsafe impl Send for LlamaContextParams {}
unsafe impl Sync for LlamaContextParams {}

impl LlamaContextParams {
    /// Set the side of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let params = params.with_n_ctx(NonZeroU32::new(2048));
    /// assert_eq!(params.n_ctx(), NonZeroU32::new(2048));
    /// ```
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.context_params.n_ctx = n_ctx.map_or(0, NonZeroU32::get);
        self
    }

    /// Get the size of the context.
    ///
    /// [`None`] if the context size is specified by the model and not the context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(512));
    #[must_use]
    pub const fn n_ctx(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.context_params.n_ctx)
    }

    /// Set the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_batch(2048);
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub const fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.context_params.n_batch = n_batch;
        self
    }

    /// Get the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub const fn n_batch(&self) -> u32 {
        self.context_params.n_batch
    }

    /// Set the `n_ubatch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_ubatch(512);
    /// assert_eq!(params.n_ubatch(), 512);
    /// ```
    #[must_use]
    pub const fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.context_params.n_ubatch = n_ubatch;
        self
    }

    /// Get the `n_ubatch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_ubatch(), 512);
    /// ```
    #[must_use]
    pub const fn n_ubatch(&self) -> u32 {
        self.context_params.n_ubatch
    }

    /// Set the flash attention policy using llama.cpp enum
    #[must_use]
    pub const fn with_flash_attention_policy(
        mut self,
        policy: llama_cpp_bindings_sys::llama_flash_attn_type,
    ) -> Self {
        self.context_params.flash_attn_type = policy;
        self
    }

    /// Get the flash attention policy
    #[must_use]
    pub const fn flash_attention_policy(&self) -> llama_cpp_bindings_sys::llama_flash_attn_type {
        self.context_params.flash_attn_type
    }

    /// Set the `offload_kqv` parameter to control offloading KV cache & KQV ops to GPU
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_offload_kqv(false);
    /// assert_eq!(params.offload_kqv(), false);
    /// ```
    #[must_use]
    pub const fn with_offload_kqv(mut self, enabled: bool) -> Self {
        self.context_params.offload_kqv = enabled;
        self
    }

    /// Get the `offload_kqv` parameter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.offload_kqv(), true);
    /// ```
    #[must_use]
    pub const fn offload_kqv(&self) -> bool {
        self.context_params.offload_kqv
    }

    /// Set the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::{LlamaContextParams, RopeScalingType};
    /// let params = LlamaContextParams::default()
    ///     .with_rope_scaling_type(RopeScalingType::Linear);
    /// assert_eq!(params.rope_scaling_type(), RopeScalingType::Linear);
    /// ```
    #[must_use]
    pub fn with_rope_scaling_type(mut self, rope_scaling_type: RopeScalingType) -> Self {
        self.context_params.rope_scaling_type = i32::from(rope_scaling_type);
        self
    }

    /// Get the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_scaling_type(), llama_cpp_bindings::context::params::RopeScalingType::Unspecified);
    /// ```
    #[must_use]
    pub fn rope_scaling_type(&self) -> RopeScalingType {
        RopeScalingType::from(self.context_params.rope_scaling_type)
    }

    /// Set the rope frequency base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_rope_freq_base(0.5);
    /// assert_eq!(params.rope_freq_base(), 0.5);
    /// ```
    #[must_use]
    pub const fn with_rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.context_params.rope_freq_base = rope_freq_base;
        self
    }

    /// Get the rope frequency base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_base(), 0.0);
    /// ```
    #[must_use]
    pub const fn rope_freq_base(&self) -> f32 {
        self.context_params.rope_freq_base
    }

    /// Set the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///   .with_rope_freq_scale(0.5);
    /// assert_eq!(params.rope_freq_scale(), 0.5);
    /// ```
    #[must_use]
    pub const fn with_rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.context_params.rope_freq_scale = rope_freq_scale;
        self
    }

    /// Get the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_scale(), 0.0);
    /// ```
    #[must_use]
    pub const fn rope_freq_scale(&self) -> f32 {
        self.context_params.rope_freq_scale
    }

    /// Get the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads(), 4);
    /// ```
    #[must_use]
    pub const fn n_threads(&self) -> i32 {
        self.context_params.n_threads
    }

    /// Get the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads_batch(), 4);
    /// ```
    #[must_use]
    pub const fn n_threads_batch(&self) -> i32 {
        self.context_params.n_threads_batch
    }

    /// Set the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads(8);
    /// assert_eq!(params.n_threads(), 8);
    /// ```
    #[must_use]
    pub const fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads = n_threads;
        self
    }

    /// Set the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads_batch(8);
    /// assert_eq!(params.n_threads_batch(), 8);
    /// ```
    #[must_use]
    pub const fn with_n_threads_batch(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads_batch = n_threads;
        self
    }

    /// Check whether embeddings are enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert!(!params.embeddings());
    /// ```
    #[must_use]
    pub const fn embeddings(&self) -> bool {
        self.context_params.embeddings
    }

    /// Enable the use of embeddings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_embeddings(true);
    /// assert!(params.embeddings());
    /// ```
    #[must_use]
    pub const fn with_embeddings(mut self, embedding: bool) -> Self {
        self.context_params.embeddings = embedding;
        self
    }

    /// Set the evaluation callback.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern "C" fn cb_eval_fn(
    ///     t: *mut llama_cpp_bindings_sys::ggml_tensor,
    ///     ask: bool,
    ///     user_data: *mut std::ffi::c_void,
    /// ) -> bool {
    ///     false
    /// }
    ///
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_cb_eval(Some(cb_eval_fn));
    /// ```
    #[must_use]
    pub fn with_cb_eval(
        mut self,
        cb_eval: llama_cpp_bindings_sys::ggml_backend_sched_eval_callback,
    ) -> Self {
        self.context_params.cb_eval = cb_eval;
        self
    }

    /// Set the evaluation callback user data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let user_data = std::ptr::null_mut();
    /// let params = params.with_cb_eval_user_data(user_data);
    /// ```
    #[must_use]
    pub const fn with_cb_eval_user_data(
        mut self,
        cb_eval_user_data: *mut std::ffi::c_void,
    ) -> Self {
        self.context_params.cb_eval_user_data = cb_eval_user_data;
        self
    }

    /// Set the type of pooling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::{LlamaContextParams, LlamaPoolingType};
    /// let params = LlamaContextParams::default()
    ///     .with_pooling_type(LlamaPoolingType::Last);
    /// assert_eq!(params.pooling_type(), LlamaPoolingType::Last);
    /// ```
    #[must_use]
    pub fn with_pooling_type(mut self, pooling_type: LlamaPoolingType) -> Self {
        self.context_params.pooling_type = i32::from(pooling_type);
        self
    }

    /// Get the type of pooling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.pooling_type(), llama_cpp_bindings::context::params::LlamaPoolingType::Unspecified);
    /// ```
    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        LlamaPoolingType::from(self.context_params.pooling_type)
    }

    /// Set whether to use full sliding window attention
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_swa_full(false);
    /// assert_eq!(params.swa_full(), false);
    /// ```
    #[must_use]
    pub const fn with_swa_full(mut self, enabled: bool) -> Self {
        self.context_params.swa_full = enabled;
        self
    }

    /// Get whether full sliding window attention is enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.swa_full(), true);
    /// ```
    #[must_use]
    pub const fn swa_full(&self) -> bool {
        self.context_params.swa_full
    }

    /// Set the max number of sequences (i.e. distinct states for recurrent models)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_seq_max(64);
    /// assert_eq!(params.n_seq_max(), 64);
    /// ```
    #[must_use]
    pub const fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.context_params.n_seq_max = n_seq_max;
        self
    }

    /// Get the max number of sequences (i.e. distinct states for recurrent models)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_seq_max(), 1);
    /// ```
    #[must_use]
    pub const fn n_seq_max(&self) -> u32 {
        self.context_params.n_seq_max
    }
    /// Set the KV cache data type for K
    /// use `llama_cpp_bindings::context::params::{LlamaContextParams`, `KvCacheType`};
    /// let params = `LlamaContextParams::default().with_type_k(KvCacheType::Q4_0)`;
    /// `assert_eq!(params.type_k()`, `KvCacheType::Q4_0`);
    /// ```
    #[must_use]
    pub fn with_type_k(mut self, type_k: KvCacheType) -> Self {
        self.context_params.type_k = type_k.into();
        self
    }

    /// Get the KV cache data type for K
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// let _ = params.type_k();
    /// ```
    #[must_use]
    pub fn type_k(&self) -> KvCacheType {
        KvCacheType::from(self.context_params.type_k)
    }

    /// Set the KV cache data type for V
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::{LlamaContextParams, KvCacheType};
    /// let params = LlamaContextParams::default().with_type_v(KvCacheType::Q4_1);
    /// assert_eq!(params.type_v(), KvCacheType::Q4_1);
    /// ```
    #[must_use]
    pub fn with_type_v(mut self, type_v: KvCacheType) -> Self {
        self.context_params.type_v = type_v.into();
        self
    }

    /// Get the KV cache data type for V
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// let _ = params.type_v();
    /// ```
    #[must_use]
    pub fn type_v(&self) -> KvCacheType {
        KvCacheType::from(self.context_params.type_v)
    }

    /// Set the attention type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::{LlamaContextParams, LlamaAttentionType};
    /// let params = LlamaContextParams::default()
    ///     .with_attention_type(LlamaAttentionType::NonCausal);
    /// assert_eq!(params.attention_type(), LlamaAttentionType::NonCausal);
    /// ```
    #[must_use]
    pub fn with_attention_type(mut self, attention_type: LlamaAttentionType) -> Self {
        self.context_params.attention_type = i32::from(attention_type);
        self
    }

    /// Get the attention type
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_bindings::context::params::LlamaContextParams::default();
    /// assert_eq!(params.attention_type(), llama_cpp_bindings::context::params::LlamaAttentionType::Unspecified);
    /// ```
    #[must_use]
    pub fn attention_type(&self) -> LlamaAttentionType {
        LlamaAttentionType::from(self.context_params.attention_type)
    }

    /// Set the `YaRN` extrapolation factor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_yarn_ext_factor(1.0);
    /// assert!((params.yarn_ext_factor() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[must_use]
    pub const fn with_yarn_ext_factor(mut self, yarn_ext_factor: f32) -> Self {
        self.context_params.yarn_ext_factor = yarn_ext_factor;
        self
    }

    /// Get the `YaRN` extrapolation factor
    #[must_use]
    pub const fn yarn_ext_factor(&self) -> f32 {
        self.context_params.yarn_ext_factor
    }

    /// Set the `YaRN` attention factor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_yarn_attn_factor(2.0);
    /// assert!((params.yarn_attn_factor() - 2.0).abs() < f32::EPSILON);
    /// ```
    #[must_use]
    pub const fn with_yarn_attn_factor(mut self, yarn_attn_factor: f32) -> Self {
        self.context_params.yarn_attn_factor = yarn_attn_factor;
        self
    }

    /// Get the `YaRN` attention factor
    #[must_use]
    pub const fn yarn_attn_factor(&self) -> f32 {
        self.context_params.yarn_attn_factor
    }

    /// Set the `YaRN` low correction dim
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_yarn_beta_fast(32.0);
    /// assert!((params.yarn_beta_fast() - 32.0).abs() < f32::EPSILON);
    /// ```
    #[must_use]
    pub const fn with_yarn_beta_fast(mut self, yarn_beta_fast: f32) -> Self {
        self.context_params.yarn_beta_fast = yarn_beta_fast;
        self
    }

    /// Get the `YaRN` low correction dim
    #[must_use]
    pub const fn yarn_beta_fast(&self) -> f32 {
        self.context_params.yarn_beta_fast
    }

    /// Set the `YaRN` high correction dim
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_yarn_beta_slow(1.0);
    /// assert!((params.yarn_beta_slow() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[must_use]
    pub const fn with_yarn_beta_slow(mut self, yarn_beta_slow: f32) -> Self {
        self.context_params.yarn_beta_slow = yarn_beta_slow;
        self
    }

    /// Get the `YaRN` high correction dim
    #[must_use]
    pub const fn yarn_beta_slow(&self) -> f32 {
        self.context_params.yarn_beta_slow
    }

    /// Set the `YaRN` original context size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_yarn_orig_ctx(4096);
    /// assert_eq!(params.yarn_orig_ctx(), 4096);
    /// ```
    #[must_use]
    pub const fn with_yarn_orig_ctx(mut self, yarn_orig_ctx: u32) -> Self {
        self.context_params.yarn_orig_ctx = yarn_orig_ctx;
        self
    }

    /// Get the `YaRN` original context size
    #[must_use]
    pub const fn yarn_orig_ctx(&self) -> u32 {
        self.context_params.yarn_orig_ctx
    }

    /// Set the KV cache defragmentation threshold
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_defrag_thold(0.1);
    /// assert!((params.defrag_thold() - 0.1).abs() < f32::EPSILON);
    /// ```
    #[must_use]
    pub const fn with_defrag_thold(mut self, defrag_thold: f32) -> Self {
        self.context_params.defrag_thold = defrag_thold;
        self
    }

    /// Get the KV cache defragmentation threshold
    #[must_use]
    pub const fn defrag_thold(&self) -> f32 {
        self.context_params.defrag_thold
    }

    /// Set whether performance timings are disabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_no_perf(true);
    /// assert!(params.no_perf());
    /// ```
    #[must_use]
    pub const fn with_no_perf(mut self, no_perf: bool) -> Self {
        self.context_params.no_perf = no_perf;
        self
    }

    /// Get whether performance timings are disabled
    #[must_use]
    pub const fn no_perf(&self) -> bool {
        self.context_params.no_perf
    }

    /// Set whether to offload ops to GPU
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_op_offload(false);
    /// assert!(!params.op_offload());
    /// ```
    #[must_use]
    pub const fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.context_params.op_offload = op_offload;
        self
    }

    /// Get whether ops are offloaded to GPU
    #[must_use]
    pub const fn op_offload(&self) -> bool {
        self.context_params.op_offload
    }

    /// Set whether to use a unified KV cache buffer across input sequences
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_bindings::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_kv_unified(true);
    /// assert!(params.kv_unified());
    /// ```
    #[must_use]
    pub const fn with_kv_unified(mut self, kv_unified: bool) -> Self {
        self.context_params.kv_unified = kv_unified;
        self
    }

    /// Get whether a unified KV cache buffer is used across input sequences
    #[must_use]
    pub const fn kv_unified(&self) -> bool {
        self.context_params.kv_unified
    }
}

/// Default parameters for `LlamaContext`. (as defined in llama.cpp by `llama_context_default_params`)
/// ```
/// # use std::num::NonZeroU32;
/// use llama_cpp_bindings::context::params::{LlamaContextParams, RopeScalingType};
/// let params = LlamaContextParams::default();
/// assert_eq!(params.n_ctx(), NonZeroU32::new(512), "n_ctx should be 512");
/// assert_eq!(params.rope_scaling_type(), RopeScalingType::Unspecified);
/// ```
impl Default for LlamaContextParams {
    fn default() -> Self {
        let context_params = unsafe { llama_cpp_bindings_sys::llama_context_default_params() };
        Self { context_params }
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCacheType, LlamaAttentionType, LlamaPoolingType, RopeScalingType};

    #[test]
    fn rope_scaling_type_unknown_defaults_to_unspecified() {
        assert_eq!(RopeScalingType::from(99), RopeScalingType::Unspecified);
        assert_eq!(RopeScalingType::from(-100), RopeScalingType::Unspecified);
    }

    #[test]
    fn pooling_type_unknown_defaults_to_unspecified() {
        assert_eq!(LlamaPoolingType::from(99), LlamaPoolingType::Unspecified);
        assert_eq!(LlamaPoolingType::from(-50), LlamaPoolingType::Unspecified);
    }

    #[test]
    fn kv_cache_type_unknown_preserves_raw_value() {
        let unknown_raw: llama_cpp_bindings_sys::ggml_type = 99999;
        let cache_type = KvCacheType::from(unknown_raw);

        assert_eq!(cache_type, KvCacheType::Unknown(99999));

        let back: llama_cpp_bindings_sys::ggml_type = cache_type.into();

        assert_eq!(back, 99999);
    }

    #[test]
    fn default_params_have_expected_values() {
        let params = super::LlamaContextParams::default();

        assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(512));
        assert_eq!(params.n_batch(), 2048);
        assert_eq!(params.n_ubatch(), 512);
        assert_eq!(params.rope_scaling_type(), RopeScalingType::Unspecified);
        assert_eq!(params.pooling_type(), LlamaPoolingType::Unspecified);
    }

    #[test]
    fn with_n_ctx_sets_value() {
        let params =
            super::LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(2048));

        assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(2048));
    }

    #[test]
    fn with_n_ctx_none_sets_zero() {
        let params = super::LlamaContextParams::default().with_n_ctx(None);

        assert_eq!(params.n_ctx(), None);
    }

    #[test]
    fn with_n_batch_sets_value() {
        let params = super::LlamaContextParams::default().with_n_batch(4096);

        assert_eq!(params.n_batch(), 4096);
    }

    #[test]
    fn with_n_ubatch_sets_value() {
        let params = super::LlamaContextParams::default().with_n_ubatch(1024);

        assert_eq!(params.n_ubatch(), 1024);
    }

    #[test]
    fn with_n_seq_max_sets_value() {
        let params = super::LlamaContextParams::default().with_n_seq_max(64);

        assert_eq!(params.n_seq_max(), 64);
    }

    #[test]
    fn with_embeddings_enables() {
        let params = super::LlamaContextParams::default().with_embeddings(true);

        assert!(params.embeddings());
    }

    #[test]
    fn with_embeddings_disables() {
        let params = super::LlamaContextParams::default().with_embeddings(false);

        assert!(!params.embeddings());
    }

    #[test]
    fn with_offload_kqv_disables() {
        let params = super::LlamaContextParams::default().with_offload_kqv(false);

        assert!(!params.offload_kqv());
    }

    #[test]
    fn with_offload_kqv_enables() {
        let params = super::LlamaContextParams::default().with_offload_kqv(true);

        assert!(params.offload_kqv());
    }

    #[test]
    fn with_swa_full_disables() {
        let params = super::LlamaContextParams::default().with_swa_full(false);

        assert!(!params.swa_full());
    }

    #[test]
    fn with_swa_full_enables() {
        let params = super::LlamaContextParams::default().with_swa_full(true);

        assert!(params.swa_full());
    }

    #[test]
    fn with_rope_scaling_type_linear() {
        let params =
            super::LlamaContextParams::default().with_rope_scaling_type(RopeScalingType::Linear);

        assert_eq!(params.rope_scaling_type(), RopeScalingType::Linear);
    }

    #[test]
    fn with_rope_scaling_type_yarn() {
        let params =
            super::LlamaContextParams::default().with_rope_scaling_type(RopeScalingType::Yarn);

        assert_eq!(params.rope_scaling_type(), RopeScalingType::Yarn);
    }

    #[test]
    fn with_rope_scaling_type_none() {
        let params =
            super::LlamaContextParams::default().with_rope_scaling_type(RopeScalingType::None);

        assert_eq!(params.rope_scaling_type(), RopeScalingType::None);
    }

    #[test]
    fn with_rope_freq_base_sets_value() {
        let params = super::LlamaContextParams::default().with_rope_freq_base(10000.0);

        assert!((params.rope_freq_base() - 10000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_rope_freq_scale_sets_value() {
        let params = super::LlamaContextParams::default().with_rope_freq_scale(0.5);

        assert!((params.rope_freq_scale() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn with_n_threads_sets_value() {
        let params = super::LlamaContextParams::default().with_n_threads(16);

        assert_eq!(params.n_threads(), 16);
    }

    #[test]
    fn with_n_threads_batch_sets_value() {
        let params = super::LlamaContextParams::default().with_n_threads_batch(16);

        assert_eq!(params.n_threads_batch(), 16);
    }

    #[test]
    fn with_pooling_type_mean() {
        let params = super::LlamaContextParams::default().with_pooling_type(LlamaPoolingType::Mean);

        assert_eq!(params.pooling_type(), LlamaPoolingType::Mean);
    }

    #[test]
    fn with_pooling_type_cls() {
        let params = super::LlamaContextParams::default().with_pooling_type(LlamaPoolingType::Cls);

        assert_eq!(params.pooling_type(), LlamaPoolingType::Cls);
    }

    #[test]
    fn with_pooling_type_last() {
        let params = super::LlamaContextParams::default().with_pooling_type(LlamaPoolingType::Last);

        assert_eq!(params.pooling_type(), LlamaPoolingType::Last);
    }

    #[test]
    fn with_pooling_type_rank() {
        let params = super::LlamaContextParams::default().with_pooling_type(LlamaPoolingType::Rank);

        assert_eq!(params.pooling_type(), LlamaPoolingType::Rank);
    }

    #[test]
    fn with_pooling_type_none() {
        let params = super::LlamaContextParams::default().with_pooling_type(LlamaPoolingType::None);

        assert_eq!(params.pooling_type(), LlamaPoolingType::None);
    }

    #[test]
    fn with_type_k_sets_value() {
        let params = super::LlamaContextParams::default().with_type_k(KvCacheType::Q4_0);

        assert_eq!(params.type_k(), KvCacheType::Q4_0);
    }

    #[test]
    fn with_type_v_sets_value() {
        let params = super::LlamaContextParams::default().with_type_v(KvCacheType::Q4_1);

        assert_eq!(params.type_v(), KvCacheType::Q4_1);
    }

    #[test]
    fn with_flash_attention_policy_sets_value() {
        let params = super::LlamaContextParams::default()
            .with_flash_attention_policy(llama_cpp_bindings_sys::LLAMA_FLASH_ATTN_TYPE_ENABLED);

        assert_eq!(
            params.flash_attention_policy(),
            llama_cpp_bindings_sys::LLAMA_FLASH_ATTN_TYPE_ENABLED
        );
    }

    #[test]
    fn builder_chaining_preserves_all_values() {
        let params = super::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(1024))
            .with_n_batch(4096)
            .with_n_ubatch(256)
            .with_n_threads(8)
            .with_n_threads_batch(12)
            .with_embeddings(true)
            .with_offload_kqv(false)
            .with_rope_scaling_type(RopeScalingType::Yarn)
            .with_rope_freq_base(5000.0)
            .with_rope_freq_scale(0.25);

        assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(1024));
        assert_eq!(params.n_batch(), 4096);
        assert_eq!(params.n_ubatch(), 256);
        assert_eq!(params.n_threads(), 8);
        assert_eq!(params.n_threads_batch(), 12);
        assert!(params.embeddings());
        assert!(!params.offload_kqv());
        assert_eq!(params.rope_scaling_type(), RopeScalingType::Yarn);
        assert!((params.rope_freq_base() - 5000.0).abs() < f32::EPSILON);
        assert!((params.rope_freq_scale() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn rope_scaling_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, RopeScalingType::Unspecified),
            (0, RopeScalingType::None),
            (1, RopeScalingType::Linear),
            (2, RopeScalingType::Yarn),
        ] {
            let from_raw = RopeScalingType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }

    #[test]
    fn pooling_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, LlamaPoolingType::Unspecified),
            (0, LlamaPoolingType::None),
            (1, LlamaPoolingType::Mean),
            (2, LlamaPoolingType::Cls),
            (3, LlamaPoolingType::Last),
            (4, LlamaPoolingType::Rank),
        ] {
            let from_raw = LlamaPoolingType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }

    #[test]
    fn kv_cache_type_all_known_variants_roundtrip() {
        let all_variants = [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::Q4_0,
            KvCacheType::Q4_1,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
            KvCacheType::Q8_0,
            KvCacheType::Q8_1,
            KvCacheType::Q2_K,
            KvCacheType::Q3_K,
            KvCacheType::Q4_K,
            KvCacheType::Q5_K,
            KvCacheType::Q6_K,
            KvCacheType::Q8_K,
            KvCacheType::IQ2_XXS,
            KvCacheType::IQ2_XS,
            KvCacheType::IQ3_XXS,
            KvCacheType::IQ1_S,
            KvCacheType::IQ4_NL,
            KvCacheType::IQ3_S,
            KvCacheType::IQ2_S,
            KvCacheType::IQ4_XS,
            KvCacheType::I8,
            KvCacheType::I16,
            KvCacheType::I32,
            KvCacheType::I64,
            KvCacheType::F64,
            KvCacheType::IQ1_M,
            KvCacheType::BF16,
            KvCacheType::TQ1_0,
            KvCacheType::TQ2_0,
            KvCacheType::MXFP4,
        ];

        for variant in all_variants {
            let ggml_type: llama_cpp_bindings_sys::ggml_type = variant.into();
            let back = KvCacheType::from(ggml_type);

            assert_eq!(back, variant);
        }
    }

    #[test]
    fn with_cb_eval_sets_callback() {
        extern "C" fn test_cb_eval(
            _tensor: *mut llama_cpp_bindings_sys::ggml_tensor,
            _ask: bool,
            _user_data: *mut std::ffi::c_void,
        ) -> bool {
            false
        }

        let result = test_cb_eval(std::ptr::null_mut(), false, std::ptr::null_mut());

        assert!(!result);

        let params = super::LlamaContextParams::default().with_cb_eval(Some(test_cb_eval));

        assert!(params.context_params.cb_eval.is_some());
    }

    #[test]
    fn with_cb_eval_user_data_sets_pointer() {
        let mut value: i32 = 42;
        let user_data = (&raw mut value).cast::<std::ffi::c_void>();
        let params = super::LlamaContextParams::default().with_cb_eval_user_data(user_data);

        assert_eq!(params.context_params.cb_eval_user_data, user_data);
    }

    #[test]
    fn with_flash_attention_policy_disabled() {
        let params = super::LlamaContextParams::default()
            .with_flash_attention_policy(llama_cpp_bindings_sys::LLAMA_FLASH_ATTN_TYPE_DISABLED);

        assert_eq!(
            params.flash_attention_policy(),
            llama_cpp_bindings_sys::LLAMA_FLASH_ATTN_TYPE_DISABLED
        );
    }

    #[test]
    fn attention_type_unknown_defaults_to_unspecified() {
        assert_eq!(
            LlamaAttentionType::from(99),
            LlamaAttentionType::Unspecified
        );
        assert_eq!(
            LlamaAttentionType::from(-50),
            LlamaAttentionType::Unspecified
        );
    }

    #[test]
    fn attention_type_roundtrip_all_variants() {
        for (raw, expected) in [
            (-1, LlamaAttentionType::Unspecified),
            (0, LlamaAttentionType::Causal),
            (1, LlamaAttentionType::NonCausal),
        ] {
            let from_raw = LlamaAttentionType::from(raw);
            assert_eq!(from_raw, expected);

            let back_to_raw: i32 = from_raw.into();
            assert_eq!(back_to_raw, raw);
        }
    }

    #[test]
    fn with_attention_type_causal() {
        let params =
            super::LlamaContextParams::default().with_attention_type(LlamaAttentionType::Causal);

        assert_eq!(params.attention_type(), LlamaAttentionType::Causal);
    }

    #[test]
    fn with_attention_type_non_causal() {
        let params =
            super::LlamaContextParams::default().with_attention_type(LlamaAttentionType::NonCausal);

        assert_eq!(params.attention_type(), LlamaAttentionType::NonCausal);
    }

    #[test]
    fn with_yarn_ext_factor_sets_value() {
        let params = super::LlamaContextParams::default().with_yarn_ext_factor(1.5);

        assert!((params.yarn_ext_factor() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn with_yarn_attn_factor_sets_value() {
        let params = super::LlamaContextParams::default().with_yarn_attn_factor(2.0);

        assert!((params.yarn_attn_factor() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_yarn_beta_fast_sets_value() {
        let params = super::LlamaContextParams::default().with_yarn_beta_fast(32.0);

        assert!((params.yarn_beta_fast() - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_yarn_beta_slow_sets_value() {
        let params = super::LlamaContextParams::default().with_yarn_beta_slow(1.0);

        assert!((params.yarn_beta_slow() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_yarn_orig_ctx_sets_value() {
        let params = super::LlamaContextParams::default().with_yarn_orig_ctx(4096);

        assert_eq!(params.yarn_orig_ctx(), 4096);
    }

    #[test]
    fn with_defrag_thold_sets_value() {
        let params = super::LlamaContextParams::default().with_defrag_thold(0.1);

        assert!((params.defrag_thold() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn with_no_perf_enables() {
        let params = super::LlamaContextParams::default().with_no_perf(true);

        assert!(params.no_perf());
    }

    #[test]
    fn with_no_perf_disables() {
        let params = super::LlamaContextParams::default().with_no_perf(false);

        assert!(!params.no_perf());
    }

    #[test]
    fn with_op_offload_enables() {
        let params = super::LlamaContextParams::default().with_op_offload(true);

        assert!(params.op_offload());
    }

    #[test]
    fn with_op_offload_disables() {
        let params = super::LlamaContextParams::default().with_op_offload(false);

        assert!(!params.op_offload());
    }

    #[test]
    fn with_kv_unified_enables() {
        let params = super::LlamaContextParams::default().with_kv_unified(true);

        assert!(params.kv_unified());
    }

    #[test]
    fn with_kv_unified_disables() {
        let params = super::LlamaContextParams::default().with_kv_unified(false);

        assert!(!params.kv_unified());
    }
}
