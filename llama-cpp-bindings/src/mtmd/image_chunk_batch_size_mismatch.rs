#[derive(Debug, PartialEq, Eq)]
pub struct ImageChunkBatchSizeMismatch {
    pub image_tokens: usize,
    pub n_batch: i32,
}
