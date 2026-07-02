use std::fmt::Debug;
use std::num::NonZeroU32;

use crate::context::kv_cache_type::KvCacheType;
use crate::context::llama_attention_type::LlamaAttentionType;
use crate::context::llama_pooling_type::LlamaPoolingType;
use crate::context::rope_scaling_type::RopeScalingType;

#[derive(Debug, Clone, Copy)]
#[expect(
    missing_docs,
    reason = "field meanings mirror llama.cpp's `llama_context_params` C struct; restating each \
              one inline would risk drift from the upstream spec — the doc-comment on the struct \
              points at the canonical reference"
)]
pub struct LlamaContextParams {
    pub context_params: llama_cpp_bindings_sys::llama_context_params,
}

unsafe impl Send for LlamaContextParams {}
unsafe impl Sync for LlamaContextParams {}

impl LlamaContextParams {
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.context_params.n_ctx = n_ctx.map_or(0, NonZeroU32::get);
        self
    }

    #[must_use]
    pub const fn n_ctx(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.context_params.n_ctx)
    }

    #[must_use]
    pub const fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.context_params.n_batch = n_batch;
        self
    }

    #[must_use]
    pub const fn n_batch(&self) -> u32 {
        self.context_params.n_batch
    }

    #[must_use]
    pub const fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.context_params.n_ubatch = n_ubatch;
        self
    }

    #[must_use]
    pub const fn n_ubatch(&self) -> u32 {
        self.context_params.n_ubatch
    }

    #[must_use]
    pub const fn with_flash_attention_policy(
        mut self,
        policy: llama_cpp_bindings_sys::llama_flash_attn_type,
    ) -> Self {
        self.context_params.flash_attn_type = policy;
        self
    }

    #[must_use]
    pub const fn flash_attention_policy(&self) -> llama_cpp_bindings_sys::llama_flash_attn_type {
        self.context_params.flash_attn_type
    }

    #[must_use]
    pub const fn with_offload_kqv(mut self, enabled: bool) -> Self {
        self.context_params.offload_kqv = enabled;
        self
    }

    #[must_use]
    pub const fn offload_kqv(&self) -> bool {
        self.context_params.offload_kqv
    }

    #[must_use]
    pub fn with_rope_scaling_type(mut self, rope_scaling_type: RopeScalingType) -> Self {
        self.context_params.rope_scaling_type = i32::from(rope_scaling_type);
        self
    }

    #[must_use]
    pub fn rope_scaling_type(&self) -> RopeScalingType {
        RopeScalingType::from(self.context_params.rope_scaling_type)
    }

    #[must_use]
    pub const fn with_rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.context_params.rope_freq_base = rope_freq_base;
        self
    }

    #[must_use]
    pub const fn rope_freq_base(&self) -> f32 {
        self.context_params.rope_freq_base
    }

    #[must_use]
    pub const fn with_rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.context_params.rope_freq_scale = rope_freq_scale;
        self
    }

    #[must_use]
    pub const fn rope_freq_scale(&self) -> f32 {
        self.context_params.rope_freq_scale
    }

    #[must_use]
    pub const fn n_threads(&self) -> i32 {
        self.context_params.n_threads
    }

    #[must_use]
    pub const fn n_threads_batch(&self) -> i32 {
        self.context_params.n_threads_batch
    }

    #[must_use]
    pub const fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads = n_threads;
        self
    }

    #[must_use]
    pub const fn with_n_threads_batch(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads_batch = n_threads;
        self
    }

    #[must_use]
    pub const fn embeddings(&self) -> bool {
        self.context_params.embeddings
    }

    #[must_use]
    pub const fn with_embeddings(mut self, embedding: bool) -> Self {
        self.context_params.embeddings = embedding;
        self
    }

    #[must_use]
    pub fn with_cb_eval(
        mut self,
        cb_eval: llama_cpp_bindings_sys::ggml_backend_sched_eval_callback,
    ) -> Self {
        self.context_params.cb_eval = cb_eval;
        self
    }

    #[must_use]
    pub const fn with_cb_eval_user_data(
        mut self,
        cb_eval_user_data: *mut std::ffi::c_void,
    ) -> Self {
        self.context_params.cb_eval_user_data = cb_eval_user_data;
        self
    }

    #[must_use]
    pub fn with_pooling_type(mut self, pooling_type: LlamaPoolingType) -> Self {
        self.context_params.pooling_type = i32::from(pooling_type);
        self
    }

    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        LlamaPoolingType::from(self.context_params.pooling_type)
    }

    #[must_use]
    pub const fn with_swa_full(mut self, enabled: bool) -> Self {
        self.context_params.swa_full = enabled;
        self
    }

    #[must_use]
    pub const fn swa_full(&self) -> bool {
        self.context_params.swa_full
    }

    #[must_use]
    pub const fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.context_params.n_seq_max = n_seq_max;
        self
    }

    #[must_use]
    pub const fn n_seq_max(&self) -> u32 {
        self.context_params.n_seq_max
    }
    #[must_use]
    pub fn with_type_k(mut self, type_k: KvCacheType) -> Self {
        self.context_params.type_k = type_k.into();
        self
    }

    #[must_use]
    pub fn type_k(&self) -> KvCacheType {
        KvCacheType::from(self.context_params.type_k)
    }

    #[must_use]
    pub fn with_type_v(mut self, type_v: KvCacheType) -> Self {
        self.context_params.type_v = type_v.into();
        self
    }

    #[must_use]
    pub fn type_v(&self) -> KvCacheType {
        KvCacheType::from(self.context_params.type_v)
    }

    #[must_use]
    pub fn with_attention_type(mut self, attention_type: LlamaAttentionType) -> Self {
        self.context_params.attention_type = i32::from(attention_type);
        self
    }

    #[must_use]
    pub fn attention_type(&self) -> LlamaAttentionType {
        LlamaAttentionType::from(self.context_params.attention_type)
    }

    #[must_use]
    pub const fn with_yarn_ext_factor(mut self, yarn_ext_factor: f32) -> Self {
        self.context_params.yarn_ext_factor = yarn_ext_factor;
        self
    }

    #[must_use]
    pub const fn yarn_ext_factor(&self) -> f32 {
        self.context_params.yarn_ext_factor
    }

    #[must_use]
    pub const fn with_yarn_attn_factor(mut self, yarn_attn_factor: f32) -> Self {
        self.context_params.yarn_attn_factor = yarn_attn_factor;
        self
    }

    #[must_use]
    pub const fn yarn_attn_factor(&self) -> f32 {
        self.context_params.yarn_attn_factor
    }

    #[must_use]
    pub const fn with_yarn_beta_fast(mut self, yarn_beta_fast: f32) -> Self {
        self.context_params.yarn_beta_fast = yarn_beta_fast;
        self
    }

    #[must_use]
    pub const fn yarn_beta_fast(&self) -> f32 {
        self.context_params.yarn_beta_fast
    }

    #[must_use]
    pub const fn with_yarn_beta_slow(mut self, yarn_beta_slow: f32) -> Self {
        self.context_params.yarn_beta_slow = yarn_beta_slow;
        self
    }

    #[must_use]
    pub const fn yarn_beta_slow(&self) -> f32 {
        self.context_params.yarn_beta_slow
    }

    #[must_use]
    pub const fn with_yarn_orig_ctx(mut self, yarn_orig_ctx: u32) -> Self {
        self.context_params.yarn_orig_ctx = yarn_orig_ctx;
        self
    }

    #[must_use]
    pub const fn yarn_orig_ctx(&self) -> u32 {
        self.context_params.yarn_orig_ctx
    }

    #[must_use]
    pub const fn with_defrag_thold(mut self, defrag_thold: f32) -> Self {
        self.context_params.defrag_thold = defrag_thold;
        self
    }

    #[must_use]
    pub const fn defrag_thold(&self) -> f32 {
        self.context_params.defrag_thold
    }

    #[must_use]
    pub const fn with_no_perf(mut self, no_perf: bool) -> Self {
        self.context_params.no_perf = no_perf;
        self
    }

    #[must_use]
    pub const fn no_perf(&self) -> bool {
        self.context_params.no_perf
    }

    #[must_use]
    pub const fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.context_params.op_offload = op_offload;
        self
    }

    #[must_use]
    pub const fn op_offload(&self) -> bool {
        self.context_params.op_offload
    }

    #[must_use]
    pub const fn with_kv_unified(mut self, kv_unified: bool) -> Self {
        self.context_params.kv_unified = kv_unified;
        self
    }

    #[must_use]
    pub const fn kv_unified(&self) -> bool {
        self.context_params.kv_unified
    }
}

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
