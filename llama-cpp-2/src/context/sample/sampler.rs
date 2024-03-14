//! A more rusty way of sampling. Allows for adding a stack of sampling steps and a `finalizer` which selects a token from the remaining candidates.
//!
//! # Example
//!
//! ```rust
//!
//! ```

use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

/// A series of sampling steps that will produce a token.
struct Sampler {
    steps: Vec<Box<dyn FnMut(&mut LlamaTokenDataArray) -> ()>>,
    finalizer: Box<dyn FnMut(LlamaTokenDataArray) -> Option<LlamaToken>>,
}

impl Sampler {
    /// Create a very simple sampler that selects the token with the highest probability.
    fn greedy() -> Self {
        Self {
            steps: Vec::new(),
            finalizer: Box::new(|mut token_data| {
                if token_data.data.is_empty() {
                    return None;
                }
                if token_data.sorted {
                    Some(token_data[0])
                } else {
                    token_data
                }
            }),
        }
    }
}
