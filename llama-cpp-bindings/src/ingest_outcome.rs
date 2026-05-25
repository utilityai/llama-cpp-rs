use crate::sampled_token::SampledToken;

#[derive(Clone, Debug)]
pub struct IngestOutcome {
    pub sampled_token: SampledToken,
    pub visible_piece: String,
    pub raw_piece: String,
}
