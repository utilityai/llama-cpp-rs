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
pub struct TokenBoundary {
    pub open: LlamaToken,
    pub close: LlamaToken,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct SampledTokenClassifierMarkers {
    pub reasoning: Option<TokenBoundary>,
    pub tool_call: Option<TokenBoundary>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SampledTokenClassifier {
    markers: SampledTokenClassifierMarkers,
    in_reasoning: bool,
    in_tool_call: bool,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
}

impl SampledTokenClassifier {
    #[must_use]
    pub const fn new(markers: SampledTokenClassifierMarkers) -> Self {
        Self {
            markers,
            in_reasoning: false,
            in_tool_call: false,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        }
    }

    /// Build a classifier with no marker pairs known. Every ingested token is
    /// reported as [`SampledToken::Undeterminable`].
    #[must_use]
    pub const fn undetermined() -> Self {
        Self::new(SampledTokenClassifierMarkers {
            reasoning: None,
            tool_call: None,
        })
    }

    /// Build a classifier that only knows reasoning markers. Tokens emitted
    /// outside the reasoning block are classified as [`SampledToken::Content`].
    #[must_use]
    pub const fn with_reasoning(open_token: LlamaToken, close_token: LlamaToken) -> Self {
        Self::new(SampledTokenClassifierMarkers {
            reasoning: Some(TokenBoundary {
                open: open_token,
                close: close_token,
            }),
            tool_call: None,
        })
    }

    pub fn ingest(&mut self, token: LlamaToken) -> SampledToken {
        if self.in_tool_call {
            return self.ingest_within_tool_call(token);
        }

        if self.in_reasoning {
            return self.ingest_within_reasoning(token);
        }

        if let Some(boundary) = self.markers.tool_call
            && token == boundary.open
        {
            self.in_tool_call = true;
            self.usage.record_tool_call_token();

            return SampledToken::ToolCall(token);
        }

        if let Some(boundary) = self.markers.reasoning
            && token == boundary.open
        {
            self.in_reasoning = true;
            self.usage.record_reasoning_token();

            return SampledToken::Reasoning(token);
        }

        if self.markers.reasoning.is_none() && self.markers.tool_call.is_none() {
            self.usage.record_undeterminable_token();

            return SampledToken::Undeterminable(token);
        }

        self.usage.record_content_token();

        SampledToken::Content(token)
    }

    fn ingest_within_tool_call(&mut self, token: LlamaToken) -> SampledToken {
        if let Some(boundary) = self.markers.tool_call
            && token == boundary.close
        {
            self.in_tool_call = false;
        }

        self.usage.record_tool_call_token();

        SampledToken::ToolCall(token)
    }

    fn ingest_within_reasoning(&mut self, token: LlamaToken) -> SampledToken {
        if let Some(boundary) = self.markers.reasoning
            && token == boundary.close
        {
            self.in_reasoning = false;
        }

        self.usage.record_reasoning_token();

        SampledToken::Reasoning(token)
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

    #[must_use]
    pub const fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    #[must_use]
    pub const fn is_in_tool_call(&self) -> bool {
        self.in_tool_call
    }

    #[must_use]
    pub const fn markers(&self) -> &SampledTokenClassifierMarkers {
        &self.markers
    }
}

#[cfg(test)]
mod tests {
    use super::SampledTokenClassifier;
    use super::SampledTokenClassifierMarkers;
    use super::TokenBoundary;
    use crate::error::TokenUsageError;
    use crate::llama_batch::LlamaBatch;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    const REASONING_OPEN: LlamaToken = LlamaToken::new(100);
    const REASONING_CLOSE: LlamaToken = LlamaToken::new(200);
    const TOOL_CALL_OPEN: LlamaToken = LlamaToken::new(300);
    const TOOL_CALL_CLOSE: LlamaToken = LlamaToken::new(400);

    fn fresh_reasoning_classifier() -> SampledTokenClassifier {
        SampledTokenClassifier::with_reasoning(REASONING_OPEN, REASONING_CLOSE)
    }

    fn fresh_full_classifier() -> SampledTokenClassifier {
        SampledTokenClassifier::new(SampledTokenClassifierMarkers {
            reasoning: Some(TokenBoundary {
                open: REASONING_OPEN,
                close: REASONING_CLOSE,
            }),
            tool_call: Some(TokenBoundary {
                open: TOOL_CALL_OPEN,
                close: TOOL_CALL_CLOSE,
            }),
        })
    }

    #[test]
    fn content_token_outside_blocks_classified_as_content() {
        let mut classifier = fresh_full_classifier();
        let token = LlamaToken::new(1);

        assert_eq!(classifier.ingest(token), SampledToken::Content(token));
    }

    #[test]
    fn reasoning_open_enters_reasoning_state() {
        let mut classifier = fresh_full_classifier();

        assert_eq!(
            classifier.ingest(REASONING_OPEN),
            SampledToken::Reasoning(REASONING_OPEN)
        );
        assert!(classifier.is_in_reasoning());
    }

    #[test]
    fn reasoning_close_exits_reasoning_state() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(REASONING_OPEN);

        assert_eq!(
            classifier.ingest(REASONING_CLOSE),
            SampledToken::Reasoning(REASONING_CLOSE)
        );
        assert!(!classifier.is_in_reasoning());
    }

    #[test]
    fn tool_call_open_enters_tool_call_state() {
        let mut classifier = fresh_full_classifier();

        assert_eq!(
            classifier.ingest(TOOL_CALL_OPEN),
            SampledToken::ToolCall(TOOL_CALL_OPEN)
        );
        assert!(classifier.is_in_tool_call());
    }

    #[test]
    fn tool_call_close_exits_tool_call_state() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(TOOL_CALL_OPEN);

        assert_eq!(
            classifier.ingest(TOOL_CALL_CLOSE),
            SampledToken::ToolCall(TOOL_CALL_CLOSE)
        );
        assert!(!classifier.is_in_tool_call());
    }

    #[test]
    fn token_inside_tool_call_classified_as_tool_call() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(TOOL_CALL_OPEN);
        let inner = LlamaToken::new(42);

        assert_eq!(classifier.ingest(inner), SampledToken::ToolCall(inner));
    }

    #[test]
    fn reasoning_marker_inside_tool_call_stays_tool_call() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(TOOL_CALL_OPEN);

        assert_eq!(
            classifier.ingest(REASONING_OPEN),
            SampledToken::ToolCall(REASONING_OPEN)
        );
        assert!(classifier.is_in_tool_call());
        assert!(!classifier.is_in_reasoning());
    }

    #[test]
    fn tool_call_marker_inside_reasoning_stays_reasoning() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(REASONING_OPEN);

        assert_eq!(
            classifier.ingest(TOOL_CALL_OPEN),
            SampledToken::Reasoning(TOOL_CALL_OPEN)
        );
        assert!(classifier.is_in_reasoning());
        assert!(!classifier.is_in_tool_call());
    }

    #[test]
    fn markers_getter_returns_constructor_input() {
        let markers = SampledTokenClassifierMarkers {
            reasoning: Some(TokenBoundary {
                open: REASONING_OPEN,
                close: REASONING_CLOSE,
            }),
            tool_call: Some(TokenBoundary {
                open: TOOL_CALL_OPEN,
                close: TOOL_CALL_CLOSE,
            }),
        };
        let classifier = SampledTokenClassifier::new(markers);

        assert_eq!(*classifier.markers(), markers);
    }

    #[test]
    fn undetermined_classifier_reports_no_markers() {
        let classifier = SampledTokenClassifier::undetermined();

        assert_eq!(classifier.markers().reasoning, None);
        assert_eq!(classifier.markers().tool_call, None);
    }

    #[test]
    fn classifier_with_only_reasoning_emits_content_outside_block() {
        let mut classifier = fresh_reasoning_classifier();
        let token = LlamaToken::new(1);

        assert_eq!(classifier.ingest(token), SampledToken::Content(token));
        assert_eq!(
            classifier.ingest(REASONING_OPEN),
            SampledToken::Reasoning(REASONING_OPEN)
        );
        assert_eq!(
            classifier.ingest(REASONING_CLOSE),
            SampledToken::Reasoning(REASONING_CLOSE)
        );
        assert_eq!(
            classifier.ingest(LlamaToken::new(7)),
            SampledToken::Content(LlamaToken::new(7))
        );
    }

    #[test]
    fn classifier_without_markers_emits_undeterminable() {
        let mut classifier = SampledTokenClassifier::undetermined();

        assert_eq!(
            classifier.ingest(REASONING_OPEN),
            SampledToken::Undeterminable(REASONING_OPEN)
        );
        assert_eq!(
            classifier.ingest(TOOL_CALL_OPEN),
            SampledToken::Undeterminable(TOOL_CALL_OPEN)
        );
    }

    #[test]
    fn ingest_records_tool_call_in_usage() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(TOOL_CALL_OPEN);
        classifier.ingest(LlamaToken::new(5));
        classifier.ingest(LlamaToken::new(6));
        classifier.ingest(TOOL_CALL_CLOSE);

        assert_eq!(classifier.usage().tool_call_tokens(), 4);
        assert_eq!(classifier.usage().content_tokens(), 0);
        assert_eq!(classifier.usage().reasoning_tokens(), 0);
    }

    #[test]
    fn ingest_records_reasoning_in_usage() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(REASONING_OPEN);
        classifier.ingest(LlamaToken::new(5));
        classifier.ingest(REASONING_CLOSE);

        assert_eq!(classifier.usage().reasoning_tokens(), 3);
        assert_eq!(classifier.usage().tool_call_tokens(), 0);
        assert_eq!(classifier.usage().content_tokens(), 0);
    }

    #[test]
    fn ingest_records_content_in_usage() {
        let mut classifier = fresh_full_classifier();
        classifier.ingest(LlamaToken::new(1));
        classifier.ingest(LlamaToken::new(2));

        assert_eq!(classifier.usage().content_tokens(), 2);
    }

    #[test]
    fn record_prompt_tokens_updates_usage() {
        let mut classifier = fresh_reasoning_classifier();
        classifier.record_prompt_tokens(11);
        classifier.record_prompt_tokens(2);

        assert_eq!(classifier.usage().prompt_tokens(), 13);
    }

    #[test]
    fn record_cached_prompt_tokens_updates_usage() {
        let mut classifier = fresh_reasoning_classifier();
        classifier.record_prompt_tokens(10);
        classifier.record_cached_prompt_tokens(4).unwrap();

        assert_eq!(classifier.usage().cached_prompt_tokens(), 4);
    }

    #[test]
    fn record_cached_above_prompt_returns_error() {
        let mut classifier = fresh_reasoning_classifier();
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
    fn into_usage_returns_accumulated_counters() {
        let mut classifier = fresh_full_classifier();
        classifier.record_prompt_tokens(5);
        classifier.ingest(LlamaToken::new(1));
        classifier.ingest(REASONING_OPEN);
        classifier.ingest(REASONING_CLOSE);
        classifier.ingest(TOOL_CALL_OPEN);
        classifier.ingest(TOOL_CALL_CLOSE);

        let usage = classifier.into_usage();

        assert_eq!(usage.prompt_tokens(), 5);
        assert_eq!(usage.content_tokens(), 1);
        assert_eq!(usage.reasoning_tokens(), 2);
        assert_eq!(usage.tool_call_tokens(), 2);
        assert_eq!(usage.completion_tokens(), 5);
    }

    #[test]
    fn feed_prompt_to_batch_stages_one_pending_on_success() {
        let mut classifier = fresh_reasoning_classifier();
        let mut batch = LlamaBatch::new(4, 1).unwrap();

        classifier
            .feed_prompt_to_batch(&mut batch, LlamaToken::new(1), 0, &[0], false)
            .unwrap();

        assert_eq!(classifier.pending_prompt_tokens(), 1);
        assert_eq!(classifier.usage().prompt_tokens(), 0);
    }

    #[test]
    fn commit_prompt_tokens_moves_pending_into_committed() {
        let mut classifier = fresh_reasoning_classifier();
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
    fn discard_pending_prompt_tokens_resets_pending_without_touching_usage() {
        let mut classifier = fresh_reasoning_classifier();
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
    fn feed_prompt_to_batch_does_not_stage_when_batch_rejects() {
        let mut classifier = fresh_reasoning_classifier();
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
    fn feed_prompt_sequence_to_batch_does_not_stage_full_count_when_batch_rejects() {
        let mut classifier = fresh_reasoning_classifier();
        let mut batch = LlamaBatch::new(2, 1).unwrap();
        let tokens = [LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)];

        let rejection = classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false);

        assert!(rejection.is_err());
        assert_eq!(classifier.pending_prompt_tokens(), 0);
    }
}
