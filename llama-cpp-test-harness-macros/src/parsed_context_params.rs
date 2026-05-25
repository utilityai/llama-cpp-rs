#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsedContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads_batch: Option<i32>,
    pub embeddings: bool,
}
