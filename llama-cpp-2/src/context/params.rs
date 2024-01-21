//! A safe wrapper around `llama_context_params`.
use llama_cpp_sys_2;
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
impl From<i8> for RopeScalingType {
    fn from(value: i8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Linear,
            2 => Self::Yarn,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `RopeScalingType`.
impl From<RopeScalingType> for i8 {
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
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.context_params.n_ctx = n_ctx.map_or(0, |n_ctx| n_ctx.get());
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
    pub fn n_ctx(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.context_params.n_ctx)
    }

    /// Get the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_2::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_scaling_type(), llama_cpp_2::context::params::RopeScalingType::Unspecified);
    /// ```
    pub fn rope_scaling_type(&self) -> RopeScalingType {
        RopeScalingType::from(self.context_params.rope_scaling_type)
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
        Self { context_params, }
    }
}
