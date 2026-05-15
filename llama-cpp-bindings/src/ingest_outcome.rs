use crate::sampled_token::SampledToken;

#[derive(Clone, Debug)]
pub struct IngestOutcome {
    pub sampled_token: SampledToken,
    /// Empty when the token is part of a recognised marker boundary; otherwise
    /// the decoded UTF-8 piece. Callers should stream `visible_piece` and skip
    /// emission when it is empty.
    pub visible_piece: String,
    /// Always the decoded UTF-8 piece, even for marker-boundary tokens. Useful
    /// for accumulating the full raw model output (e.g. for downstream parser
    /// cross-checks) without losing marker bytes.
    pub raw_piece: String,
}
