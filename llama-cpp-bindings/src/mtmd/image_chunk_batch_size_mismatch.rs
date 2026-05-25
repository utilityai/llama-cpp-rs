#[derive(Debug)]
pub struct ImageChunkBatchSizeMismatch {
    pub image_tokens: u32,
    pub n_batch: u32,
}
