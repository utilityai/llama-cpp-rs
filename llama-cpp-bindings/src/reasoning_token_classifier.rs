//! Deterministic state machine that classifies sampled tokens as reasoning, content, or undeterminable.

use crate::sampled_token::SampledToken;
use crate::token::LlamaToken;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct ReasoningBoundary {
    open: LlamaToken,
    close: LlamaToken,
}

/// Classifies each sampled `LlamaToken` as either reasoning, content, or
/// undeterminable, based on the model's detected reasoning marker token ids.
///
/// When constructed via [`ReasoningTokenClassifier::new`] with explicit boundary
/// token ids, the classifier emits [`SampledToken::Reasoning`] for the open
/// token, every token between, and the close token, and
/// [`SampledToken::Content`] for everything else.
///
/// When constructed via [`ReasoningTokenClassifier::undetermined`], every input
/// token is emitted as [`SampledToken::Undeterminable`].
///
/// Pathological model output (close before open, repeated open/close, etc.) is
/// handled deterministically — no panics.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReasoningTokenClassifier {
    boundary: Option<ReasoningBoundary>,
    in_reasoning: bool,
}

impl ReasoningTokenClassifier {
    /// Create a new classifier from the resolved boundary token ids.
    #[must_use]
    pub const fn new(open_token: LlamaToken, close_token: LlamaToken) -> Self {
        Self {
            boundary: Some(ReasoningBoundary {
                open: open_token,
                close: close_token,
            }),
            in_reasoning: false,
        }
    }

    /// Create a classifier that treats every input token as
    /// [`SampledToken::Undeterminable`]. Used for models whose reasoning
    /// markers could not be detected.
    #[must_use]
    pub const fn undetermined() -> Self {
        Self {
            boundary: None,
            in_reasoning: false,
        }
    }

    /// Classify `token` and advance the internal state.
    pub fn classify(&mut self, token: LlamaToken) -> SampledToken {
        let Some(boundary) = self.boundary else {
            return SampledToken::Undeterminable(token);
        };

        if self.in_reasoning {
            if token == boundary.close {
                self.in_reasoning = false;
            }

            SampledToken::Reasoning(token)
        } else if token == boundary.open {
            self.in_reasoning = true;

            SampledToken::Reasoning(token)
        } else {
            SampledToken::Content(token)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReasoningTokenClassifier;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    const OPEN: LlamaToken = LlamaToken::new(100);
    const CLOSE: LlamaToken = LlamaToken::new(200);

    fn fresh_classifier() -> ReasoningTokenClassifier {
        ReasoningTokenClassifier::new(OPEN, CLOSE)
    }

    #[test]
    fn content_token_outside_reasoning_classified_as_content() {
        let mut classifier = fresh_classifier();
        let token = LlamaToken::new(1);

        assert_eq!(classifier.classify(token), SampledToken::Content(token));
    }

    #[test]
    fn open_token_emits_reasoning_and_enters_reasoning_state() {
        let mut classifier = fresh_classifier();

        assert_eq!(classifier.classify(OPEN), SampledToken::Reasoning(OPEN));
        let after_open = LlamaToken::new(1);
        assert_eq!(
            classifier.classify(after_open),
            SampledToken::Reasoning(after_open)
        );
    }

    #[test]
    fn token_inside_reasoning_classified_as_reasoning() {
        let mut classifier = fresh_classifier();
        classifier.classify(OPEN);
        let inner = LlamaToken::new(42);

        assert_eq!(classifier.classify(inner), SampledToken::Reasoning(inner));
    }

    #[test]
    fn close_token_emits_reasoning_and_exits_reasoning_state() {
        let mut classifier = fresh_classifier();
        classifier.classify(OPEN);

        assert_eq!(classifier.classify(CLOSE), SampledToken::Reasoning(CLOSE));
        let after_close = LlamaToken::new(7);
        assert_eq!(
            classifier.classify(after_close),
            SampledToken::Content(after_close)
        );
    }

    #[test]
    fn token_after_close_classified_as_content() {
        let mut classifier = fresh_classifier();
        classifier.classify(OPEN);
        classifier.classify(LlamaToken::new(5));
        classifier.classify(CLOSE);
        let after = LlamaToken::new(9);

        assert_eq!(classifier.classify(after), SampledToken::Content(after));
    }

    #[test]
    fn multiple_reasoning_blocks_alternate_correctly() {
        let mut classifier = fresh_classifier();
        let regular = LlamaToken::new(1);
        let inner = LlamaToken::new(2);

        assert_eq!(classifier.classify(regular), SampledToken::Content(regular));
        assert_eq!(classifier.classify(OPEN), SampledToken::Reasoning(OPEN));
        assert_eq!(classifier.classify(inner), SampledToken::Reasoning(inner));
        assert_eq!(classifier.classify(CLOSE), SampledToken::Reasoning(CLOSE));
        assert_eq!(classifier.classify(regular), SampledToken::Content(regular));
        assert_eq!(classifier.classify(OPEN), SampledToken::Reasoning(OPEN));
        assert_eq!(classifier.classify(CLOSE), SampledToken::Reasoning(CLOSE));
        assert_eq!(classifier.classify(regular), SampledToken::Content(regular));
    }

    #[test]
    fn close_token_outside_reasoning_classified_as_content() {
        let mut classifier = fresh_classifier();

        assert_eq!(classifier.classify(CLOSE), SampledToken::Content(CLOSE));
        let next = LlamaToken::new(3);
        assert_eq!(classifier.classify(next), SampledToken::Content(next));
    }

    #[test]
    fn open_token_while_already_in_reasoning_stays_in_reasoning() {
        let mut classifier = fresh_classifier();
        classifier.classify(OPEN);

        assert_eq!(classifier.classify(OPEN), SampledToken::Reasoning(OPEN));
        let inner = LlamaToken::new(4);
        assert_eq!(classifier.classify(inner), SampledToken::Reasoning(inner));
    }

    #[test]
    fn undetermined_classifier_emits_undeterminable_for_every_input() {
        let mut classifier = ReasoningTokenClassifier::undetermined();

        assert_eq!(
            classifier.classify(OPEN),
            SampledToken::Undeterminable(OPEN)
        );
        assert_eq!(
            classifier.classify(CLOSE),
            SampledToken::Undeterminable(CLOSE)
        );
        let other = LlamaToken::new(7);
        assert_eq!(
            classifier.classify(other),
            SampledToken::Undeterminable(other)
        );
    }
}
