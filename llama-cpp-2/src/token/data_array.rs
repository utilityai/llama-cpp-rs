//! an rusty equivalent of `llama_token_data`.
use crate::token::data::LlamaTokenData;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// is the data sorted?
    pub sorted: bool,
}
