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

/// A safe wrapper around `llama_context_params`.
///
/// Generally this should be created with [`Default::default()`] and then modified with `with_*` methods.
///
/// # Examples
///
/// ```rust
/// # use std::num::NonZeroU32;
/// use llama_cpp_2::context::params::LlamaContextParams;
///
///let ctx_params = LlamaContextParams::default()
///    .with_n_ctx(NonZeroU32::new(2048))
///    .with_seed(1234);
///
/// assert_eq!(ctx_params.seed(), 1234);
/// assert_eq!(ctx_params.n_ctx(), NonZeroU32::new(2048));
/// ```
#[derive(Debug, Clone)]
#[allow(
    missing_docs,
    clippy::struct_excessive_bools,
    clippy::module_name_repetitions
)]
pub struct LlamaContextParams {
    pub(crate) context_params: llama_cpp_sys_2::llama_context_params,
}

/// SAFETY: we do not currently allow setting or reading the pointers that cause this to not be automatically send or sync.
unsafe impl Send for LlamaContextParams {}
unsafe impl Sync for LlamaContextParams {}

impl LlamaContextParams {
    /// Set the seed of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let params = params.with_seed(1234);
    /// assert_eq!(params.seed(), 1234);
    /// ```
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.context_params.seed = seed;
        self
    }

    /// Get the seed of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_seed(1234);
    /// assert_eq!(params.seed(), 1234);
    /// ```
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.context_params.seed
    }

    /// Set the side of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let params = params.with_n_ctx(NonZeroU32::new(2048));
    /// assert_eq!(params.n_ctx(), NonZeroU32::new(2048));
    /// ```
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.context_params.n_ctx = n_ctx.map_or(0, std::num::NonZeroU32::get);
        self
    }

    /// Get the size of the context.
    ///
    /// [`None`] if the context size is specified by the model and not the context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(512));
    #[must_use]
    pub fn n_ctx(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.context_params.n_ctx)
    }

    /// Set the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_batch(2048);
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.context_params.n_batch = n_batch;
        self
    }

    /// Get the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        self.context_params.n_batch
    }

    /// Set the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::{LlamaContextParams, RopeScalingType};
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
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_scaling_type(), llama_cpp_2::context::params::RopeScalingType::Unspecified);
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
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_rope_freq_base(0.5);
    /// assert_eq!(params.rope_freq_base(), 0.5);
    /// ```
    #[must_use]
    pub fn with_rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.context_params.rope_freq_base = rope_freq_base;
        self
    }

    /// Get the rope frequency base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_base(), 0.0);
    /// ```
    #[must_use]
    pub fn rope_freq_base(&self) -> f32 {
        self.context_params.rope_freq_base
    }

    /// Set the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///   .with_rope_freq_scale(0.5);
    /// assert_eq!(params.rope_freq_scale(), 0.5);
    /// ```
    #[must_use]
    pub fn with_rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.context_params.rope_freq_scale = rope_freq_scale;
        self
    }

    /// Get the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_scale(), 0.0);
    /// ```
    #[must_use]
    pub fn rope_freq_scale(&self) -> f32 {
        self.context_params.rope_freq_scale
    }

    /// Get the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads(), 4);
    /// ```
    #[must_use]
    pub fn n_threads(&self) -> i32 {
        self.context_params.n_threads
    }

    /// Get the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads_batch(), 4);
    /// ```
    #[must_use]
    pub fn n_threads_batch(&self) -> i32 {
        self.context_params.n_threads_batch
    }

    /// Set the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads(8);
    /// assert_eq!(params.n_threads(), 8);
    /// ```
    #[must_use]
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads = n_threads;
        self
    }

    /// Set the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads_batch(8);
    /// assert_eq!(params.n_threads_batch(), 8);
    /// ```
    #[must_use]
    pub fn with_n_threads_batch(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads_batch = n_threads;
        self
    }

    /// Check whether embeddings are enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert!(!params.embeddings());
    /// ```
    #[must_use]
    pub fn embeddings(&self) -> bool {
        self.context_params.embeddings
    }

    /// Enable the use of embeddings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_embeddings(true);
    /// assert!(params.embeddings());
    /// ```
    #[must_use]
    pub fn with_embeddings(mut self, embedding: bool) -> Self {
        self.context_params.embeddings = embedding;
        self
    }

    /// Set the evaluation callback.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern "C" fn cb_eval_fn(
    ///     t: *mut llama_cpp_sys_2::ggml_tensor,
    ///     ask: bool,
    ///     user_data: *mut std::ffi::c_void,
    /// ) -> bool {
    ///     false
    /// }
    ///
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// params.with_cb_eval(Some(cb_eval_fn));
    /// ```
    pub fn with_cb_eval(
        mut self,
        cb_eval: llama_cpp_sys_2::ggml_backend_sched_eval_callback,
    ) -> Self {
        self.context_params.cb_eval = cb_eval;
        self
    }

    /// Set the evaluation callback user data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llama_cpp_2::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let user_data = std::ptr::null_mut();
    /// params.with_cb_eval_user_data(user_data);
    /// ```
    pub fn with_cb_eval_user_data(mut self, cb_eval_user_data: *mut std::ffi::c_void) -> Self {
        self.context_params.cb_eval_user_data = cb_eval_user_data;
        self
    }
}

/// Default parameters for `LlamaContext`. (as defined in llama.cpp by `llama_context_default_params`)
/// ```
/// # use std::num::NonZeroU32;
/// use llama_cpp_2::context::params::{LlamaContextParams, RopeScalingType};
/// let params = LlamaContextParams::default();
/// assert_eq!(params.n_ctx(), NonZeroU32::new(512), "n_ctx should be 512");
/// assert_eq!(params.rope_scaling_type(), RopeScalingType::Unspecified);
/// ```
impl Default for LlamaContextParams {
    fn default() -> Self {
        let context_params = unsafe { llama_cpp_sys_2::llama_context_default_params() };
        Self { context_params }
    }
}
