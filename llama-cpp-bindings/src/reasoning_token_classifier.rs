use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_sys::llama_seq_id;

use crate::context::LlamaContext;
use crate::error::EvalMultimodalChunksError;
use crate::error::SampleError;
use crate::error::TokenUsageError;
use crate::llama_batch::BatchAddError;
use crate::llama_batch::LlamaBatch;
use crate::mtmd::MtmdContext;
use crate::mtmd::MtmdInputChunkType;
use crate::mtmd::MtmdInputChunks;
use crate::sampled_token::SampledToken;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::token_usage::TokenUsage;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct ReasoningBoundary {
    open: LlamaToken,
    close: LlamaToken,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReasoningTokenClassifier {
    boundary: Option<ReasoningBoundary>,
    in_reasoning: bool,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
}

impl ReasoningTokenClassifier {
    #[must_use]
    pub const fn new(open_token: LlamaToken, close_token: LlamaToken) -> Self {
        Self {
            boundary: Some(ReasoningBoundary {
                open: open_token,
                close: close_token,
            }),
            in_reasoning: false,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        }
    }

    #[must_use]
    pub const fn undetermined() -> Self {
        Self {
            boundary: None,
            in_reasoning: false,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        }
    }

    pub fn ingest(&mut self, token: LlamaToken) -> SampledToken {
        let Some(boundary) = self.boundary else {
            self.usage.record_undeterminable_token();

            return SampledToken::Undeterminable(token);
        };

        if self.in_reasoning {
            if token == boundary.close {
                self.in_reasoning = false;
            }
            self.usage.record_reasoning_token();

            SampledToken::Reasoning(token)
        } else if token == boundary.open {
            self.in_reasoning = true;
            self.usage.record_reasoning_token();

            SampledToken::Reasoning(token)
        } else {
            self.usage.record_content_token();

            SampledToken::Content(token)
        }
    }

    /// # Errors
    /// Forwards [`LlamaSampler::sample`] errors verbatim. Nothing is recorded on failure.
    pub fn sample(
        &mut self,
        sampler: &mut LlamaSampler,
        context: &LlamaContext,
        idx: i32,
    ) -> Result<SampledToken, SampleError> {
        let raw = sampler.sample(context, idx)?;

        Ok(self.ingest(raw))
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add`] errors verbatim. Nothing is staged on failure.
    pub fn feed_prompt_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        token: LlamaToken,
        position: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        batch.add(&SampledToken::Content(token), position, seq_ids, logits)?;
        self.pending_prompt_tokens = self.pending_prompt_tokens.saturating_add(1);

        Ok(())
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add_sequence`] errors verbatim. Nothing is staged on failure.
    pub fn feed_prompt_sequence_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
        seq_id: llama_seq_id,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        batch.add_sequence(tokens, seq_id, logits_all)?;
        self.pending_prompt_tokens = self
            .pending_prompt_tokens
            .saturating_add(tokens.len() as u64);

        Ok(())
    }

    pub const fn commit_prompt_tokens(&mut self) -> u64 {
        let promoted = self.pending_prompt_tokens;
        self.usage.record_prompt_tokens(promoted);
        self.pending_prompt_tokens = 0;

        promoted
    }

    pub const fn discard_pending_prompt_tokens(&mut self) -> u64 {
        let discarded = self.pending_prompt_tokens;
        self.pending_prompt_tokens = 0;

        discarded
    }

    #[must_use]
    pub const fn pending_prompt_tokens(&self) -> u64 {
        self.pending_prompt_tokens
    }

    /// # Errors
    /// Returns [`EvalMultimodalChunksError::EvalFailed`] when the underlying
    /// `eval_chunks` call fails (no counters move),
    /// [`EvalMultimodalChunksError::UnknownChunkType`] when a chunk reports a
    /// type unknown to this binding, or
    /// [`EvalMultimodalChunksError::ChunkOutOfBounds`] when a valid index returns
    /// `None` from `chunks.get`.
    #[expect(
        clippy::too_many_arguments,
        reason = "thin wrapper over MtmdInputChunks::eval_chunks; parameter shape mirrors the underlying API"
    )]
    pub fn eval_multimodal_chunks(
        &mut self,
        chunks: &MtmdInputChunks,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        n_past: llama_pos,
        seq_id: llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_pos, EvalMultimodalChunksError> {
        let n_past_after =
            chunks.eval_chunks(mtmd_ctx, llama_ctx, n_past, seq_id, n_batch, logits_last)?;

        for index in 0..chunks.len() {
            let chunk = chunks
                .get(index)
                .ok_or(EvalMultimodalChunksError::ChunkOutOfBounds(index))?;
            let n_tokens = chunk.n_tokens() as u64;
            match chunk.chunk_type()? {
                MtmdInputChunkType::Text => self.usage.record_prompt_tokens(n_tokens),
                MtmdInputChunkType::Image => self.usage.record_input_image_tokens(n_tokens),
                MtmdInputChunkType::Audio => self.usage.record_input_audio_tokens(n_tokens),
            }
        }

        Ok(n_past_after)
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.usage.record_prompt_tokens(count);
    }

    /// # Errors
    /// Forwards [`TokenUsageError::CachedExceedsPrompt`] when the running cached total would
    /// exceed the prompt total.
    pub const fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), TokenUsageError> {
        self.usage.record_cached_prompt_tokens(count)
    }

    #[must_use]
    pub const fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    #[must_use]
    pub const fn into_usage(self) -> TokenUsage {
        self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::ReasoningTokenClassifier;
    use crate::error::TokenUsageError;
    use crate::llama_batch::LlamaBatch;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;
    use crate::token_usage::TokenUsage;

    const OPEN: LlamaToken = LlamaToken::new(100);
    const CLOSE: LlamaToken = LlamaToken::new(200);

    fn fresh_classifier() -> ReasoningTokenClassifier {
        ReasoningTokenClassifier::new(OPEN, CLOSE)
    }

    #[test]
    fn content_token_outside_reasoning_classified_as_content() {
        let mut classifier = fresh_classifier();
        let token = LlamaToken::new(1);

        assert_eq!(classifier.ingest(token), SampledToken::Content(token));
    }

    #[test]
    fn open_token_emits_reasoning_and_enters_reasoning_state() {
        let mut classifier = fresh_classifier();

        assert_eq!(classifier.ingest(OPEN), SampledToken::Reasoning(OPEN));
        let after_open = LlamaToken::new(1);
        assert_eq!(
            classifier.ingest(after_open),
            SampledToken::Reasoning(after_open)
        );
    }

    #[test]
    fn token_inside_reasoning_classified_as_reasoning() {
        let mut classifier = fresh_classifier();
        classifier.ingest(OPEN);
        let inner = LlamaToken::new(42);

        assert_eq!(classifier.ingest(inner), SampledToken::Reasoning(inner));
    }

    #[test]
    fn close_token_emits_reasoning_and_exits_reasoning_state() {
        let mut classifier = fresh_classifier();
        classifier.ingest(OPEN);

        assert_eq!(classifier.ingest(CLOSE), SampledToken::Reasoning(CLOSE));
        let after_close = LlamaToken::new(7);
        assert_eq!(
            classifier.ingest(after_close),
            SampledToken::Content(after_close)
        );
    }

    #[test]
    fn token_after_close_classified_as_content() {
        let mut classifier = fresh_classifier();
        classifier.ingest(OPEN);
        classifier.ingest(LlamaToken::new(5));
        classifier.ingest(CLOSE);
        let after = LlamaToken::new(9);

        assert_eq!(classifier.ingest(after), SampledToken::Content(after));
    }

    #[test]
    fn multiple_reasoning_blocks_alternate_correctly() {
        let mut classifier = fresh_classifier();
        let regular = LlamaToken::new(1);
        let inner = LlamaToken::new(2);

        assert_eq!(classifier.ingest(regular), SampledToken::Content(regular));
        assert_eq!(classifier.ingest(OPEN), SampledToken::Reasoning(OPEN));
        assert_eq!(classifier.ingest(inner), SampledToken::Reasoning(inner));
        assert_eq!(classifier.ingest(CLOSE), SampledToken::Reasoning(CLOSE));
        assert_eq!(classifier.ingest(regular), SampledToken::Content(regular));
        assert_eq!(classifier.ingest(OPEN), SampledToken::Reasoning(OPEN));
        assert_eq!(classifier.ingest(CLOSE), SampledToken::Reasoning(CLOSE));
        assert_eq!(classifier.ingest(regular), SampledToken::Content(regular));
    }

    #[test]
    fn close_token_outside_reasoning_classified_as_content() {
        let mut classifier = fresh_classifier();

        assert_eq!(classifier.ingest(CLOSE), SampledToken::Content(CLOSE));
        let next = LlamaToken::new(3);
        assert_eq!(classifier.ingest(next), SampledToken::Content(next));
    }

    #[test]
    fn open_token_while_already_in_reasoning_stays_in_reasoning() {
        let mut classifier = fresh_classifier();
        classifier.ingest(OPEN);

        assert_eq!(classifier.ingest(OPEN), SampledToken::Reasoning(OPEN));
        let inner = LlamaToken::new(4);
        assert_eq!(classifier.ingest(inner), SampledToken::Reasoning(inner));
    }

    #[test]
    fn undetermined_classifier_emits_undeterminable_for_every_input() {
        let mut classifier = ReasoningTokenClassifier::undetermined();

        assert_eq!(classifier.ingest(OPEN), SampledToken::Undeterminable(OPEN));
        assert_eq!(
            classifier.ingest(CLOSE),
            SampledToken::Undeterminable(CLOSE)
        );
        let other = LlamaToken::new(7);
        assert_eq!(
            classifier.ingest(other),
            SampledToken::Undeterminable(other)
        );
    }

    #[test]
    fn usage_starts_at_default_for_fresh_classifier() {
        assert_eq!(*fresh_classifier().usage(), TokenUsage::default());
        assert_eq!(
            *ReasoningTokenClassifier::undetermined().usage(),
            TokenUsage::default()
        );
    }

    #[test]
    fn ingest_records_content_in_usage() {
        let mut classifier = fresh_classifier();
        classifier.ingest(LlamaToken::new(1));
        classifier.ingest(LlamaToken::new(2));

        assert_eq!(classifier.usage().content_tokens(), 2);
        assert_eq!(classifier.usage().reasoning_tokens(), 0);
        assert_eq!(classifier.usage().undeterminable_tokens(), 0);
    }

    #[test]
    fn ingest_records_reasoning_in_usage_for_open_token_and_inner() {
        let mut classifier = fresh_classifier();
        classifier.ingest(OPEN);
        classifier.ingest(LlamaToken::new(5));
        classifier.ingest(LlamaToken::new(6));
        classifier.ingest(CLOSE);

        assert_eq!(classifier.usage().reasoning_tokens(), 4);
        assert_eq!(classifier.usage().content_tokens(), 0);
    }

    #[test]
    fn ingest_records_undeterminable_in_usage_when_no_boundary() {
        let mut classifier = ReasoningTokenClassifier::undetermined();
        classifier.ingest(LlamaToken::new(1));
        classifier.ingest(LlamaToken::new(2));
        classifier.ingest(LlamaToken::new(3));

        assert_eq!(classifier.usage().undeterminable_tokens(), 3);
        assert_eq!(classifier.usage().content_tokens(), 0);
        assert_eq!(classifier.usage().reasoning_tokens(), 0);
        assert_eq!(classifier.usage().completion_tokens(), 0);
    }

    #[test]
    fn record_prompt_tokens_updates_usage() {
        let mut classifier = fresh_classifier();
        classifier.record_prompt_tokens(11);
        classifier.record_prompt_tokens(2);

        assert_eq!(classifier.usage().prompt_tokens(), 13);
    }

    #[test]
    fn record_cached_prompt_tokens_updates_usage() {
        let mut classifier = fresh_classifier();
        classifier.record_prompt_tokens(10);
        classifier.record_cached_prompt_tokens(4).unwrap();

        assert_eq!(classifier.usage().cached_prompt_tokens(), 4);
    }

    #[test]
    fn record_cached_above_prompt_returns_error_in_classifier_too() {
        let mut classifier = fresh_classifier();
        classifier.record_prompt_tokens(2);

        let result = classifier.record_cached_prompt_tokens(3);

        assert_eq!(
            result,
            Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: 3,
                prompt: 2,
            })
        );
        assert_eq!(classifier.usage().cached_prompt_tokens(), 0);
    }

    #[test]
    fn into_usage_returns_accumulated_counters_and_consumes_classifier() {
        let mut classifier = fresh_classifier();
        classifier.record_prompt_tokens(5);
        classifier.ingest(LlamaToken::new(1));
        classifier.ingest(OPEN);
        classifier.ingest(CLOSE);

        let usage = classifier.into_usage();

        assert_eq!(usage.prompt_tokens(), 5);
        assert_eq!(usage.content_tokens(), 1);
        assert_eq!(usage.reasoning_tokens(), 2);
        assert_eq!(usage.completion_tokens(), 3);
    }

    #[test]
    fn feed_prompt_to_batch_stages_one_pending_on_success() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(4, 1).unwrap();

        classifier
            .feed_prompt_to_batch(&mut batch, LlamaToken::new(1), 0, &[0], false)
            .unwrap();

        assert_eq!(classifier.pending_prompt_tokens(), 1);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn feed_prompt_to_batch_does_not_stage_when_batch_rejects() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(1, 1).unwrap();
        classifier
            .feed_prompt_to_batch(&mut batch, LlamaToken::new(1), 0, &[0], false)
            .unwrap();

        let rejection =
            classifier.feed_prompt_to_batch(&mut batch, LlamaToken::new(2), 1, &[0], false);

        assert!(rejection.is_err());
        assert_eq!(classifier.pending_prompt_tokens(), 1);
    }

    #[test]
    fn feed_prompt_sequence_to_batch_stages_count_on_success() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(8, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];

        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)
            .unwrap();

        assert_eq!(classifier.pending_prompt_tokens(), 3);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn feed_prompt_sequence_to_batch_does_not_stage_full_count_when_batch_rejects() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(2, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];

        let rejection = classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false);

        assert!(rejection.is_err());
        assert_eq!(classifier.pending_prompt_tokens(), 0);
    }

    #[test]
    fn pending_prompt_tokens_does_not_contribute_to_prompt_or_completion() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(8, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2)];
        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)
            .unwrap();

        assert_eq!(classifier.usage().prompt_tokens(), 0);
        assert_eq!(classifier.usage().completion_tokens(), 0);
    }

    #[test]
    fn commit_prompt_tokens_moves_pending_into_committed_prompt_tokens_and_resets_pending() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(8, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)
            .unwrap();

        let promoted = classifier.commit_prompt_tokens();

        assert_eq!(promoted, 3);
        assert_eq!(classifier.pending_prompt_tokens(), 0);
        assert_eq!(classifier.usage().prompt_tokens(), 3);
    }

    #[test]
    fn commit_prompt_tokens_with_no_pending_returns_zero_and_changes_nothing() {
        let mut classifier = fresh_classifier();

        let promoted = classifier.commit_prompt_tokens();

        assert_eq!(promoted, 0);
        assert_eq!(classifier.pending_prompt_tokens(), 0);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn discard_pending_prompt_tokens_resets_pending_without_touching_usage() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(8, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2)];
        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)
            .unwrap();

        let discarded = classifier.discard_pending_prompt_tokens();

        assert_eq!(discarded, 2);
        assert_eq!(classifier.pending_prompt_tokens(), 0);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn multiple_feed_then_commit_aggregates_correctly() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        classifier
            .feed_prompt_to_batch(&mut batch, LlamaToken::new(1), 0, &[0], false)
            .unwrap();
        classifier
            .feed_prompt_sequence_to_batch(
                &mut batch,
                &[LlamaToken::new(2), LlamaToken::new(3)],
                1,
                false,
            )
            .unwrap();

        let promoted = classifier.commit_prompt_tokens();

        assert_eq!(promoted, 3);
        assert_eq!(classifier.usage().prompt_tokens(), 3);
    }

    #[test]
    fn multiple_feed_then_discard_drops_everything() {
        let mut classifier = fresh_classifier();
        let mut batch = LlamaBatch::new(16, 1).unwrap();
        classifier
            .feed_prompt_to_batch(&mut batch, LlamaToken::new(1), 0, &[0], false)
            .unwrap();
        classifier
            .feed_prompt_sequence_to_batch(
                &mut batch,
                &[LlamaToken::new(2), LlamaToken::new(3)],
                1,
                false,
            )
            .unwrap();

        let discarded = classifier.discard_pending_prompt_tokens();

        assert_eq!(discarded, 3);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn two_classifiers_sharing_a_batch_track_their_own_pending_and_committed_counts() {
        let mut request_a = fresh_classifier();
        let mut request_b = fresh_classifier();
        let mut batch = LlamaBatch::new(16, 2).unwrap();

        let tokens_a = [LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];
        let tokens_b = [LlamaToken::new(4), LlamaToken::new(5)];

        request_a
            .feed_prompt_sequence_to_batch(&mut batch, &tokens_a, 0, false)
            .unwrap();
        request_b
            .feed_prompt_sequence_to_batch(&mut batch, &tokens_b, 1, false)
            .unwrap();

        assert_eq!(request_a.pending_prompt_tokens(), 3);
        assert_eq!(request_b.pending_prompt_tokens(), 2);
        assert_eq!(request_a.usage().prompt_tokens(), 0);
        assert_eq!(request_b.usage().prompt_tokens(), 0);

        request_a.ingest(LlamaToken::new(99));

        assert_eq!(request_a.usage().content_tokens(), 1);
        assert_eq!(request_b.usage().content_tokens(), 0);

        let promoted_a = request_a.commit_prompt_tokens();
        let promoted_b = request_b.commit_prompt_tokens();

        assert_eq!(promoted_a, 3);
        assert_eq!(promoted_b, 2);
        assert_eq!(request_a.usage().prompt_tokens(), 3);
        assert_eq!(request_b.usage().prompt_tokens(), 2);
        assert_eq!(request_a.pending_prompt_tokens(), 0);
        assert_eq!(request_b.pending_prompt_tokens(), 0);
    }

    #[test]
    fn discarding_one_classifier_does_not_affect_another_sharing_the_batch() {
        let mut request_a = fresh_classifier();
        let mut request_b = fresh_classifier();
        let mut batch = LlamaBatch::new(16, 2).unwrap();

        request_a
            .feed_prompt_sequence_to_batch(
                &mut batch,
                &[LlamaToken::new(1), LlamaToken::new(2)],
                0,
                false,
            )
            .unwrap();
        request_b
            .feed_prompt_sequence_to_batch(
                &mut batch,
                &[LlamaToken::new(3), LlamaToken::new(4), LlamaToken::new(5)],
                1,
                false,
            )
            .unwrap();

        let discarded_a = request_a.discard_pending_prompt_tokens();
        let promoted_b = request_b.commit_prompt_tokens();

        assert_eq!(discarded_a, 2);
        assert_eq!(promoted_b, 3);
        assert_eq!(request_a.usage().prompt_tokens(), 0);
        assert_eq!(request_b.usage().prompt_tokens(), 3);
    }
}
