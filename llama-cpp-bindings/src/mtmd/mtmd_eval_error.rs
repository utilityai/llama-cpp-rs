use crate::mtmd::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;

/// Errors that can occur during evaluation
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// Requested batch size exceeds the context's maximum batch size
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit {
        /// The batch size requested in `eval_chunks`
        requested: i32,
        /// The maximum batch size configured on the context
        context_max: u32,
    },
    /// An image chunk's token count exceeds the per-decode `n_batch` budget,
    /// so handing it to `llama_decode` would trip the `GGML_ASSERT`.
    #[error(
        "image chunk has {} tokens but n_batch is {}",
        .0.image_tokens,
        .0.n_batch,
    )]
    ImageChunkExceedsBatchSize(ImageChunkBatchSizeMismatch),
    /// Evaluation operation failed
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}
