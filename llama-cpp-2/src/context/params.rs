//! A safe wrapper around `llama_context_params`.
use llama_cpp_sys_2::{ggml_type, llama_context_params};
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    missing_docs,
    clippy::struct_excessive_bools,
    clippy::module_name_repetitions
)]
pub struct LlamaContextParams {
    /// The random seed
    pub seed: u32,
    /// the number of tokens in the context - [`None`] if defined by the model.
    pub n_ctx: Option<NonZeroU32>,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: RopeScalingType,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub type_k: ggml_type,
    pub type_v: ggml_type,
    pub mul_mat_q: bool,
    pub logits_all: bool,
    pub embedding: bool,
    pub offload_kqv: bool,
}

/// Default parameters for `LlamaContext`. (as defined in llama.cpp by `llama_context_default_params`)
/// ```
/// # use llama_cpp_2::context::params::LlamaContextParams;
/// let params = LlamaContextParams::default();
/// assert_eq!(params.n_ctx.unwrap().get(), 512, "n_ctx should be 512");
/// ```
impl Default for LlamaContextParams {
    fn default() -> Self {
        Self::from(unsafe { llama_cpp_sys_2::llama_context_default_params() })
    }
}

impl From<llama_context_params> for LlamaContextParams {
    fn from(
        llama_context_params {
            seed,
            n_ctx,
            n_batch,
            n_threads,
            n_threads_batch,
            rope_freq_base,
            rope_freq_scale,
            type_k,
            type_v,
            mul_mat_q,
            logits_all,
            embedding,
            rope_scaling_type,
            yarn_ext_factor,
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_orig_ctx,
            offload_kqv,
        }: llama_context_params,
    ) -> Self {
        Self {
            seed,
            n_ctx: NonZeroU32::new(n_ctx),
            n_batch,
            n_threads,
            n_threads_batch,
            rope_freq_base,
            rope_freq_scale,
            type_k,
            type_v,
            mul_mat_q,
            logits_all,
            embedding,
            rope_scaling_type: RopeScalingType::from(rope_scaling_type),
            yarn_ext_factor,
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_orig_ctx,
            offload_kqv,
        }
    }
}

impl From<LlamaContextParams> for llama_context_params {
    fn from(
        LlamaContextParams {
            seed,
            n_ctx,
            n_batch,
            n_threads,
            n_threads_batch,
            rope_freq_base,
            rope_freq_scale,
            type_k,
            type_v,
            mul_mat_q,
            logits_all,
            embedding,
            rope_scaling_type,
            yarn_ext_factor,
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_orig_ctx,
            offload_kqv,
        }: LlamaContextParams,
    ) -> Self {
        llama_context_params {
            seed,
            n_ctx: n_ctx.map_or(0, NonZeroU32::get),
            n_batch,
            n_threads,
            n_threads_batch,
            rope_freq_base,
            rope_freq_scale,
            type_k,
            type_v,
            mul_mat_q,
            logits_all,
            embedding,
            rope_scaling_type: i8::from(rope_scaling_type),
            yarn_ext_factor,
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_orig_ctx,
            offload_kqv,
        }
    }
}
