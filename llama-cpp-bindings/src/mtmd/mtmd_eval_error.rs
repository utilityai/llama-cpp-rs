use crate::mtmd::image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;

#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit { requested: i32, context_max: u32 },
    #[error(
        "image chunk has {} tokens but n_batch is {}",
        .0.image_tokens,
        .0.n_batch,
    )]
    ImageChunkExceedsBatchSize(ImageChunkBatchSizeMismatch),
    #[error("Wrapper received a null mtmd-context argument")]
    NullMtmdCtxArg,
    #[error("Wrapper received a null llama-context argument")]
    NullLlamaCtxArg,
    #[error("Wrapper received a null chunk argument")]
    NullChunkArg,
    #[error("Internal wrapper invariant violated: caller did not pass an out-new-n-past pointer")]
    NullOutNewNPastArg,
    #[error("mtmd_helper_eval_chunk_single returned nonzero code: {code}")]
    VendoredReturnedNonzeroCode { code: i32 },
    #[error("Wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("mtmd_helper_eval_chunk_single threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
}
