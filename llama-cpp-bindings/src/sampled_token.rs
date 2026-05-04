//! Classification-carrying wrapper around a sampled `LlamaToken`.

use crate::token::LlamaToken;

/// A token together with its classification.
///
/// Replaces raw [`LlamaToken`] in the post-sampling flow so the variant always
/// communicates whether the token belongs to a reasoning block, regular content,
/// or comes from a model whose reasoning markers could not be detected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SampledToken {
    /// Token outside any reasoning block on a model with detected reasoning markers.
    /// Also used by callers to wrap prompt or vocab tokens explicitly known to be
    /// non-reasoning input.
    Content(LlamaToken),
    /// Token inside a reasoning block, including the opening and closing boundary tokens.
    Reasoning(LlamaToken),
    /// Token from a model whose reasoning markers could not be detected.
    Undeterminable(LlamaToken),
}
