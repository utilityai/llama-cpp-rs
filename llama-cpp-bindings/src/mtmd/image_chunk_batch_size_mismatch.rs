/// Carried by [`super::mtmd_eval_error::MtmdEvalError::ImageChunkExceedsBatchSize`].
///
/// `n_batch` is the per-decode batch budget enforced by `cparams.n_batch` in
/// llama.cpp; `image_tokens` is the number of tokens this image chunk would
/// hand to `llama_decode`. When `image_tokens > n_batch` the C-side
/// `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` would abort the process —
/// the binding refuses the call instead.
#[derive(Debug)]
pub struct ImageChunkBatchSizeMismatch {
    pub image_tokens: u32,
    pub n_batch: u32,
}
