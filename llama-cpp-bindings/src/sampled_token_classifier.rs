use std::collections::VecDeque;

use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_sys::llama_seq_id;

use llama_cpp_bindings_types::TokenUsage;
use llama_cpp_bindings_types::TokenUsageError;

use crate::context::LlamaContext;
use crate::error::EvalMultimodalChunksError;
use crate::error::SampleError;
use crate::llama_batch::BatchAddError;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::mtmd::MtmdContext;
use crate::mtmd::MtmdInputChunks;
use crate::sampled_token::SampledToken;
use crate::sampling::LlamaSampler;
use crate::streaming_json_probe::JsonProbeOutcome;
use crate::streaming_json_probe::validate_prefix;
use crate::token::LlamaToken;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SampledTokenSection {
    Pending,
    Content,
    Reasoning,
    ToolCall,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum MarkerKind {
    ReasoningOpen,
    ReasoningClose,
    ToolCallOpen,
    ToolCallClose,
}

/// Tokenized marker sequences (token IDs, not strings).
///
/// Each marker is a `Vec<LlamaToken>` of length `>= 1`; absent markers are
/// `None`. Sequence matching at every `ingest()` is by token-ID equality,
/// never by substring scanning of decoded text.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamingMarkers {
    pub reasoning_open: Option<Vec<LlamaToken>>,
    pub reasoning_close: Option<Vec<LlamaToken>>,
    pub tool_call_open: Option<Vec<LlamaToken>>,
    pub tool_call_close: Option<Vec<LlamaToken>>,
}

impl StreamingMarkers {
    const fn has_any(&self) -> bool {
        self.reasoning_open.is_some()
            || self.reasoning_close.is_some()
            || self.tool_call_open.is_some()
            || self.tool_call_close.is_some()
    }

    fn max_token_len(&self) -> usize {
        [
            self.reasoning_open.as_deref(),
            self.reasoning_close.as_deref(),
            self.tool_call_open.as_deref(),
            self.tool_call_close.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(<[LlamaToken]>::len)
        .max()
        .unwrap_or(0)
    }

    fn lookup(&self, kind: MarkerKind) -> Option<&[LlamaToken]> {
        match kind {
            MarkerKind::ReasoningOpen => self.reasoning_open.as_deref(),
            MarkerKind::ReasoningClose => self.reasoning_close.as_deref(),
            MarkerKind::ToolCallOpen => self.tool_call_open.as_deref(),
            MarkerKind::ToolCallClose => self.tool_call_close.as_deref(),
        }
    }
}

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

#[derive(Clone, Debug)]
struct PendingToken {
    token: LlamaToken,
    decoded: String,
    section: SampledTokenSection,
    is_boundary: bool,
    is_from_prompt: bool,
    is_held_for_probe: bool,
}

#[derive(Clone, Debug)]
struct JsonProbeState {
    held_text: String,
}

#[derive(Clone, Debug)]
enum ProbeMode {
    Idle,
    Active(JsonProbeState),
}

pub struct SampledTokenClassifier<'model> {
    model: &'model LlamaModel,
    markers: StreamingMarkers,
    decoder: encoding_rs::Decoder,
    pending: VecDeque<PendingToken>,
    section: SampledTokenSection,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
    probe_mode: ProbeMode,
}

impl<'model> SampledTokenClassifier<'model> {
    #[must_use]
    pub fn new(model: &'model LlamaModel, markers: StreamingMarkers) -> Self {
        Self {
            model,
            markers,
            decoder: encoding_rs::UTF_8.new_decoder(),
            pending: VecDeque::new(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
            probe_mode: ProbeMode::Idle,
        }
    }

    /// Ingest one sampled token. Returns the outcomes that have finalised this
    /// turn — typically a single outcome, occasionally zero (the classifier is
    /// holding back tokens that may yet form a marker), or several when a
    /// buffered marker prefix diverges and the held-back tokens flush.
    ///
    /// Each [`IngestOutcome`] carries both the [`SampledToken`] variant for
    /// classification and the decoded `visible_piece` for streaming. Marker
    /// boundaries get an empty `visible_piece` so their text never reaches
    /// user-visible streams.
    pub fn ingest(&mut self, token: LlamaToken) -> Vec<IngestOutcome> {
        if !self.markers.has_any() {
            self.usage.record_undeterminable_token();
            let piece = self.decode(token);
            return vec![IngestOutcome {
                sampled_token: SampledToken::Undeterminable(token),
                visible_piece: piece.clone(),
                raw_piece: piece,
            }];
        }

        let decoded = self.decode(token);
        self.pending.push_back(PendingToken {
            token,
            decoded: decoded.clone(),
            section: self.section,
            is_boundary: false,
            is_from_prompt: false,
            is_held_for_probe: false,
        });

        self.try_consume_marker_at_tail();

        let probe_was_active = matches!(self.probe_mode, ProbeMode::Active(_));
        let mut outcomes = if probe_was_active && self.section_disengages_probe() {
            self.abandon_probe()
        } else {
            self.update_probe(&decoded)
        };

        outcomes.extend(self.drain_overflow());
        outcomes
    }

    const fn section_disengages_probe(&self) -> bool {
        matches!(
            self.section,
            SampledTokenSection::ToolCall | SampledTokenSection::Reasoning
        )
    }

    /// Replay one prompt token through the marker state machine so that the
    /// section at end-of-prompt reflects the chat template's rendered tail
    /// (e.g. for Qwen3.5/3.6 with `enable_thinking=false` the prompt ends with
    /// a closed empty `<think>...</think>` block, leaving the section in
    /// `Content`; with `enable_thinking=true` it ends inside an open `<think>`,
    /// leaving the section in `Reasoning`).
    ///
    /// Prompt tokens never produce [`IngestOutcome`]s and never increment usage
    /// counters — they are not generated content.
    pub fn ingest_prompt_token(&mut self, token: LlamaToken) {
        if !self.markers.has_any() {
            return;
        }

        self.pending.push_back(PendingToken {
            token,
            decoded: String::new(),
            section: self.section,
            is_boundary: false,
            is_from_prompt: true,
            is_held_for_probe: false,
        });

        self.try_consume_marker_at_tail();
        self.drain_overflow();
    }

    pub fn ingest_prompt_tokens(&mut self, tokens: &[LlamaToken]) {
        if !self.markers.has_any() {
            return;
        }
        for &token in tokens {
            self.ingest_prompt_token(token);
        }
    }

    /// Drain every still-buffered token. Call once at end of generation (EOG)
    /// to make sure no decoded text is silently dropped. After `flush()` the
    /// classifier behaves as if freshly constructed in terms of buffer state.
    pub fn flush(&mut self) -> Vec<IngestOutcome> {
        self.probe_mode = ProbeMode::Idle;
        let mut outcomes = Vec::with_capacity(self.pending.len());
        while let Some(entry) = self.pending.pop_front() {
            if entry.is_from_prompt {
                continue;
            }
            outcomes.push(self.finalize_entry(entry));
        }
        outcomes
    }

    fn decode(&mut self, token: LlamaToken) -> String {
        match self.model.token_to_piece(
            &SampledToken::Content(token),
            &mut self.decoder,
            true,
            None,
        ) {
            Ok(piece) => piece,
            Err(detokenize_error) => {
                tracing::debug!(
                    "token_to_piece failed during classification, dropping piece: {detokenize_error}"
                );
                String::new()
            }
        }
    }

    fn try_consume_marker_at_tail(&mut self) {
        // Probe every marker in every section so the user-visible streams stay
        // free of marker text even when the model misbehaves: a stray
        // `</think>` / `<channel|>` / `[/THINK]` while in `Content` is
        // suppressed (close markers transition to Content — a no-op when
        // already there); a nested `<think>` while in `Reasoning` is also
        // suppressed (open markers keep the section in Reasoning). Without
        // this, models like Gemma 4 E4B that emit close markers without ever
        // opening leak the literal marker text into `content_stream`.
        const PROBE_KINDS: &[MarkerKind] = &[
            MarkerKind::ReasoningOpen,
            MarkerKind::ReasoningClose,
            MarkerKind::ToolCallOpen,
            MarkerKind::ToolCallClose,
        ];

        for &kind in PROBE_KINDS {
            let Some(marker) = self.markers.lookup(kind) else {
                continue;
            };
            if marker.is_empty() || self.pending.len() < marker.len() {
                continue;
            }
            let span_start = self.pending.len() - marker.len();
            let matches = self
                .pending
                .iter()
                .skip(span_start)
                .zip(marker)
                .all(|(entry, marker_token)| entry.token == *marker_token);
            if matches {
                self.mark_marker_span(span_start, kind);
                return;
            }
        }
    }

    fn mark_marker_span(&mut self, span_start: usize, kind: MarkerKind) {
        let next_section = match kind {
            MarkerKind::ReasoningOpen => SampledTokenSection::Reasoning,
            MarkerKind::ReasoningClose | MarkerKind::ToolCallClose => SampledTokenSection::Content,
            MarkerKind::ToolCallOpen => SampledTokenSection::ToolCall,
        };
        // For open markers, the boundary tokens are classified as the destination
        // section — they are the marker itself (`<think>` is part of reasoning,
        // `<tool_call>` is part of the tool-call protocol). For close markers,
        // the boundary tokens are classified as the section the model was in:
        // a normal `</think>` while in `Reasoning` is still reasoning, but a
        // spurious `</think>` while in `Content` (e.g. some Gemma variants
        // re-emit close markers without ever opening) is just noise in the
        // content section — counting it as `Reasoning` would inflate
        // `observed_reasoning` and falsely indicate the model thought.
        let span_section = match kind {
            MarkerKind::ReasoningOpen => SampledTokenSection::Reasoning,
            MarkerKind::ToolCallOpen => SampledTokenSection::ToolCall,
            MarkerKind::ReasoningClose => {
                if self.section == SampledTokenSection::Reasoning {
                    SampledTokenSection::Reasoning
                } else {
                    SampledTokenSection::Content
                }
            }
            MarkerKind::ToolCallClose => {
                if self.section == SampledTokenSection::ToolCall {
                    SampledTokenSection::ToolCall
                } else {
                    SampledTokenSection::Content
                }
            }
        };

        for entry in self.pending.iter_mut().skip(span_start) {
            entry.is_boundary = true;
            entry.section = span_section;
        }

        self.section = next_section;
    }

    fn drain_overflow(&mut self) -> Vec<IngestOutcome> {
        let lookback = self.markers.max_token_len().saturating_sub(1);
        let mut outcomes = Vec::new();

        loop {
            let Some(front) = self.pending.front() else {
                break;
            };
            if front.is_held_for_probe {
                break;
            }
            let probe_held = self
                .pending
                .iter()
                .filter(|entry| entry.is_held_for_probe)
                .count();
            let drainable = self.pending.len().saturating_sub(probe_held);
            let beyond_lookback = drainable > lookback;
            if !front.is_boundary && !beyond_lookback {
                break;
            }
            let Some(entry) = self.pending.pop_front() else {
                break;
            };
            if entry.is_from_prompt {
                continue;
            }
            outcomes.push(self.finalize_entry(entry));
        }

        outcomes
    }

    fn update_probe(&mut self, piece: &str) -> Vec<IngestOutcome> {
        let probe_active = matches!(self.probe_mode, ProbeMode::Active(_));
        if !probe_active {
            if !self.section_allows_probe_engagement() {
                return Vec::new();
            }
            if !piece.trim_start().starts_with('{') {
                return Vec::new();
            }
            if let Some(entry) = self.pending.back_mut() {
                entry.is_held_for_probe = true;
            }
            self.probe_mode = ProbeMode::Active(JsonProbeState {
                held_text: piece.to_owned(),
            });
            return self.evaluate_probe();
        }

        if let Some(entry) = self.pending.back_mut() {
            entry.is_held_for_probe = true;
        }
        if let ProbeMode::Active(state) = &mut self.probe_mode {
            state.held_text.push_str(piece);
        }
        self.evaluate_probe()
    }

    const fn section_allows_probe_engagement(&self) -> bool {
        matches!(
            self.section,
            SampledTokenSection::Content | SampledTokenSection::Pending
        )
    }

    fn evaluate_probe(&mut self) -> Vec<IngestOutcome> {
        let outcome = match &self.probe_mode {
            ProbeMode::Active(state) => validate_prefix(&state.held_text),
            ProbeMode::Idle => return Vec::new(),
        };
        match outcome {
            JsonProbeOutcome::StillPossiblyValid => Vec::new(),
            JsonProbeOutcome::CompletedValid => self.commit_probe_as_tool_call(),
            JsonProbeOutcome::Failed => self.abandon_probe(),
        }
    }

    fn commit_probe_as_tool_call(&mut self) -> Vec<IngestOutcome> {
        if !matches!(self.probe_mode, ProbeMode::Active(_)) {
            return Vec::new();
        }
        self.probe_mode = ProbeMode::Idle;
        self.section = SampledTokenSection::Content;

        let drained: Vec<_> = self.pending.drain(..).collect();
        let mut outcomes = Vec::new();
        for mut entry in drained {
            if entry.is_held_for_probe {
                entry.section = SampledTokenSection::ToolCall;
                entry.is_held_for_probe = false;
                if !entry.is_from_prompt {
                    outcomes.push(self.finalize_entry(entry));
                }
            } else {
                self.pending.push_back(entry);
            }
        }
        outcomes
    }

    fn abandon_probe(&mut self) -> Vec<IngestOutcome> {
        if !matches!(self.probe_mode, ProbeMode::Active(_)) {
            return Vec::new();
        }
        self.probe_mode = ProbeMode::Idle;

        let drained: Vec<_> = self.pending.drain(..).collect();
        let mut outcomes = Vec::new();
        for mut entry in drained {
            if entry.is_held_for_probe {
                entry.is_held_for_probe = false;
                if !entry.is_from_prompt {
                    outcomes.push(self.finalize_entry(entry));
                }
            } else {
                self.pending.push_back(entry);
            }
        }
        outcomes
    }

    fn finalize_entry(&mut self, entry: PendingToken) -> IngestOutcome {
        let section = entry.section;
        match section {
            SampledTokenSection::Reasoning => self.usage.record_reasoning_token(),
            SampledTokenSection::Content => self.usage.record_content_token(),
            SampledTokenSection::ToolCall => self.usage.record_tool_call_token(),
            SampledTokenSection::Pending => self.usage.record_undeterminable_token(),
        }

        let sampled_token = match section {
            SampledTokenSection::Reasoning => SampledToken::Reasoning(entry.token),
            SampledTokenSection::Content => SampledToken::Content(entry.token),
            SampledTokenSection::ToolCall => SampledToken::ToolCall(entry.token),
            SampledTokenSection::Pending => SampledToken::Undeterminable(entry.token),
        };

        let visible_piece = if entry.is_boundary {
            String::new()
        } else {
            entry.decoded.clone()
        };

        IngestOutcome {
            sampled_token,
            visible_piece,
            raw_piece: entry.decoded,
        }
    }

    /// # Errors
    /// Forwards [`LlamaSampler::sample`] errors verbatim. Nothing is recorded on failure.
    ///
    /// Returns the raw sampled token (for downstream `batch.add` / `is_eog_token`
    /// calls) alongside the outcomes that finalised this turn — see
    /// [`Self::ingest`] for buffering semantics.
    pub fn sample(
        &mut self,
        sampler: &mut LlamaSampler,
        context: &LlamaContext,
        idx: i32,
    ) -> Result<(LlamaToken, Vec<IngestOutcome>), SampleError> {
        let raw = sampler.sample(context, idx)?;
        let outcomes = self.ingest(raw);

        Ok((raw, outcomes))
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
        self.ingest_prompt_token(token);
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
        self.ingest_prompt_tokens(tokens);
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
        start_position: llama_pos,
        seq_id: llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_pos, EvalMultimodalChunksError> {
        let chunk_count = chunks.len();
        // `start_position` stays read-only; `next_position` is the loop
        // accumulator that walks forward chunk-by-chunk and is the function's
        // return value. Two locals, single responsibility each.
        let mut next_position = start_position;

        for index in 0..chunk_count {
            let chunk = chunks
                .get(index)
                .ok_or(EvalMultimodalChunksError::ChunkOutOfBounds(index))?;
            let logits_for_this_chunk = logits_last && index + 1 == chunk_count;

            next_position = chunk.eval_single(
                mtmd_ctx,
                llama_ctx,
                next_position,
                seq_id,
                n_batch,
                logits_for_this_chunk,
            )?;
            crate::ingest_prompt_chunk::ingest_prompt_chunk(self, &chunk)?;
        }

        Ok(next_position)
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.usage.record_prompt_tokens(count);
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.usage.record_input_image_tokens(count);
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.usage.record_input_audio_tokens(count);
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
    pub fn into_usage(self) -> TokenUsage {
        self.usage
    }

    #[must_use]
    pub const fn current_section(&self) -> SampledTokenSection {
        self.section
    }

    #[must_use]
    pub const fn markers(&self) -> &StreamingMarkers {
        &self.markers
    }
}

#[cfg(test)]
mod tests {
    use super::IngestOutcome;
    use super::PendingToken;
    use super::ProbeMode;
    use super::SampledTokenClassifier;
    use super::SampledTokenSection;
    use super::StreamingMarkers;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    fn markers_with(
        reasoning_open: Option<Vec<LlamaToken>>,
        reasoning_close: Option<Vec<LlamaToken>>,
    ) -> StreamingMarkers {
        StreamingMarkers {
            reasoning_open,
            reasoning_close,
            tool_call_open: None,
            tool_call_close: None,
        }
    }

    /// Builds a classifier without a real model — only safe for tests that go
    /// through `try_consume_marker_at_tail` / `drain_overflow` directly, never
    /// through `ingest()` (which calls `model.token_to_piece`).
    fn synthetic_classifier(markers: StreamingMarkers) -> SampledTokenClassifier<'static> {
        SampledTokenClassifier {
            model: unsafe { &*std::ptr::NonNull::<crate::model::LlamaModel>::dangling().as_ptr() },
            markers,
            decoder: encoding_rs::UTF_8.new_decoder(),
            pending: std::collections::VecDeque::new(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: llama_cpp_bindings_types::TokenUsage::new(),
            probe_mode: ProbeMode::Idle,
        }
    }

    fn push_pending(classifier: &mut SampledTokenClassifier<'_>, token_id: i32, decoded: &str) {
        classifier.pending.push_back(PendingToken {
            token: token(token_id),
            decoded: decoded.to_owned(),
            section: classifier.section,
            is_boundary: false,
            is_from_prompt: false,
            is_held_for_probe: false,
        });
    }

    fn push_pending_from_prompt(classifier: &mut SampledTokenClassifier<'_>, token_id: i32) {
        classifier.pending.push_back(PendingToken {
            token: token(token_id),
            decoded: String::new(),
            section: classifier.section,
            is_boundary: false,
            is_from_prompt: true,
            is_held_for_probe: false,
        });
    }

    fn push_and_probe(
        classifier: &mut SampledTokenClassifier<'_>,
        token_id: i32,
        decoded: &str,
    ) -> Vec<IngestOutcome> {
        push_pending(classifier, token_id, decoded);
        classifier.try_consume_marker_at_tail();
        let probe_was_active = matches!(classifier.probe_mode, ProbeMode::Active(_));
        let mut outcomes = if probe_was_active && classifier.section_disengages_probe() {
            classifier.abandon_probe()
        } else {
            classifier.update_probe(decoded)
        };
        outcomes.extend(classifier.drain_overflow());
        outcomes
    }

    fn outcome_pieces(outcomes: &[IngestOutcome]) -> Vec<&str> {
        outcomes.iter().map(|o| o.visible_piece.as_str()).collect()
    }

    fn outcome_sections(outcomes: &[IngestOutcome]) -> Vec<SampledTokenSection> {
        outcomes
            .iter()
            .map(|o| match o.sampled_token {
                SampledToken::Reasoning(_) => SampledTokenSection::Reasoning,
                SampledToken::Content(_) => SampledTokenSection::Content,
                SampledToken::ToolCall(_) => SampledTokenSection::ToolCall,
                SampledToken::Undeterminable(_) => SampledTokenSection::Pending,
            })
            .collect()
    }

    #[test]
    fn streaming_markers_with_no_markers_reports_none() {
        let markers = StreamingMarkers::default();
        assert!(!markers.has_any());
        assert_eq!(markers.max_token_len(), 0);
    }

    #[test]
    fn streaming_markers_max_token_len_takes_longest() {
        let markers = StreamingMarkers {
            reasoning_open: Some(vec![token(1)]),
            reasoning_close: Some(vec![token(2), token(3), token(4)]),
            tool_call_open: Some(vec![token(5), token(6)]),
            tool_call_close: None,
        };
        assert_eq!(markers.max_token_len(), 3);
    }

    #[test]
    fn single_token_close_marker_when_already_in_reasoning_emits_empty_piece_for_marker() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 7, "step");
        classifier.try_consume_marker_at_tail();
        let mut outcomes = classifier.drain_overflow();

        push_pending(&mut classifier, 200, "</think>");
        classifier.try_consume_marker_at_tail();
        outcomes.extend(classifier.drain_overflow());

        push_pending(&mut classifier, 9, "Hi");
        classifier.try_consume_marker_at_tail();
        outcomes.extend(classifier.drain_overflow());

        outcomes.extend(classifier.flush());

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Reasoning,
                SampledTokenSection::Reasoning,
                SampledTokenSection::Content,
            ],
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["step", "", "Hi"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn multi_token_close_marker_suppresses_every_marker_token() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "r"), (200, "</"), (201, "thi"), (202, "nk>"), (9, "OK")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(classifier.drain_overflow());
        }
        outcomes.extend(classifier.flush());

        assert_eq!(outcome_pieces(&outcomes), vec!["r", "", "", "", "OK"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn marker_prefix_that_diverges_does_not_suppress_buffered_tokens() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "r"), (200, "a"), (201, "b"), (300, "x")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(classifier.drain_overflow());
        }
        outcomes.extend(classifier.flush());

        assert_eq!(outcome_pieces(&outcomes), vec!["r", "a", "b", "x"]);
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Reasoning(_)))
        );
        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
    }

    #[test]
    fn open_then_close_back_to_back_emits_two_empty_pieces_around_zero_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(100, "<think>"), (200, "</think>"), (9, "Hi")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(classifier.drain_overflow());
        }
        outcomes.extend(classifier.flush());

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Reasoning,
                SampledTokenSection::Reasoning,
                SampledTokenSection::Content,
            ],
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["", "", "Hi"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn spurious_reasoning_close_in_content_section_classifies_as_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_pending(&mut classifier, 200, "</think>");
        classifier.try_consume_marker_at_tail();
        let outcomes = classifier.drain_overflow();

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn spurious_tool_call_close_in_reasoning_section_classifies_as_tool_call() {
        let markers = StreamingMarkers {
            reasoning_open: Some(vec![token(100)]),
            reasoning_close: Some(vec![token(200)]),
            tool_call_open: Some(vec![token(300)]),
            tool_call_close: Some(vec![token(400)]),
        };
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::ToolCall;

        push_pending(&mut classifier, 400, "</tool_call>");
        classifier.try_consume_marker_at_tail();
        let outcomes = classifier.drain_overflow();

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::ToolCall],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn flush_drains_remaining_pending_at_eog() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 7, "abc");
        push_pending(&mut classifier, 200, "</");
        push_pending(&mut classifier, 201, "th");

        let outcomes = classifier.flush();

        assert_eq!(outcome_pieces(&outcomes), vec!["abc", "</", "th"]);
        assert!(classifier.pending.is_empty());
    }

    #[test]
    fn no_markers_marks_each_token_undeterminable_with_visible_piece() {
        let markers = StreamingMarkers::default();
        let mut classifier = synthetic_classifier(markers);

        push_pending(&mut classifier, 1, "h");
        push_pending(&mut classifier, 2, "i");
        let outcomes = classifier.flush();

        assert_eq!(outcome_pieces(&outcomes), vec!["h", "i"]);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Pending, SampledTokenSection::Pending],
        );
    }

    #[test]
    fn ingest_prompt_tokens_without_markers_is_noop() {
        let markers = StreamingMarkers::default();
        let mut classifier = synthetic_classifier(markers);

        push_pending_from_prompt(&mut classifier, 7);
        push_pending_from_prompt(&mut classifier, 8);

        assert_eq!(classifier.section, SampledTokenSection::Pending);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_through_open_close_pair_ends_in_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_through_open_only_ends_in_reasoning() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }

        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_does_not_record_usage() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 8, 9, 200, 201, 202, 11] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }
        let drained = classifier.flush();
        assert!(drained.is_empty());

        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn prompt_token_completing_marker_with_generated_token_is_suppressed_correctly() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        for token_id in [200, 201] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }

        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
        assert_eq!(classifier.pending.len(), 2);

        classifier.pending.push_back(PendingToken {
            token: token(202),
            decoded: "k>".to_owned(),
            section: classifier.section,
            is_boundary: false,
            is_from_prompt: false,
            is_held_for_probe: false,
        });
        classifier.try_consume_marker_at_tail();
        let outcomes = classifier.drain_overflow();

        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            outcomes[0].sampled_token,
            SampledToken::Reasoning(_)
        ));
        assert_eq!(outcomes[0].visible_piece, "");
        assert_eq!(outcomes[0].raw_piece, "k>");

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 1);
        assert_eq!(classifier.usage().content_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_with_multiple_round_trips_ends_in_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        // <think> body </think> <think> body </think>
        for token_id in [100, 7, 200, 100, 8, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_initial_section_is_always_pending() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let classifier = synthetic_classifier(markers);

        assert_eq!(classifier.section, SampledTokenSection::Pending);
    }

    #[test]
    fn ingest_prompt_tokens_then_drain_for_generated_token_classifies_correctly() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        // Closed-think prompt: <think> body </think>
        for token_id in [100, 7, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            classifier.drain_overflow();
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);

        // Generated content token (not from prompt): pushed with section=Content,
        // is_from_prompt=false. drain_overflow finalises it as SampledToken::Content
        // and increments usage.content_tokens.
        classifier.pending.push_back(PendingToken {
            token: token(50),
            decoded: "hi".to_owned(),
            section: classifier.section,
            is_boundary: false,
            is_from_prompt: false,
            is_held_for_probe: false,
        });
        classifier.try_consume_marker_at_tail();
        let outcomes = classifier.drain_overflow();

        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            outcomes[0].sampled_token,
            SampledToken::Content(_)
        ));
        assert_eq!(outcomes[0].visible_piece, "hi");
        assert_eq!(classifier.usage().content_tokens, 1);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn close_marker_in_content_section_is_suppressed_as_boundary() {
        // When a misbehaving model emits a close marker (e.g. `</think>`) while
        // already in the Content section, the classifier must treat it as a
        // boundary so the marker text never reaches the user-visible content
        // stream. The boundary token is classified as Content (not Reasoning):
        // there is no reasoning to close, the close marker is just noise in
        // the content section. This is the architectural backstop against
        // models that re-emit close markers without a preceding open.
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "hi"), (200, "</think>"), (8, "ok")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(classifier.drain_overflow());
        }
        outcomes.extend(classifier.flush());

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Content,
                SampledTokenSection::Content,
                SampledTokenSection::Content,
            ],
        );
        // The close marker's `visible_piece` is empty (boundary), so the
        // user-visible content stream is "hi" + "" + "ok" = "hiok".
        assert_eq!(outcome_pieces(&outcomes), vec!["hi", "", "ok"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn open_marker_in_reasoning_section_is_suppressed_as_boundary() {
        // A nested `<think>` while already in Reasoning is suppressed (so the
        // user never sees the marker text in the reasoning stream) and the
        // section stays Reasoning.
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "step1"), (100, "<think>"), (8, "step2")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(classifier.drain_overflow());
        }
        outcomes.extend(classifier.flush());

        assert_eq!(outcome_pieces(&outcomes), vec!["step1", "", "step2"]);
        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
    }

    #[test]
    fn record_prompt_tokens_updates_usage() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);

        classifier.record_prompt_tokens(7);

        assert_eq!(classifier.usage().prompt_tokens, 7);
    }

    #[test]
    fn record_cached_prompt_tokens_updates_usage_when_under_limit() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(10);

        classifier.record_cached_prompt_tokens(3).unwrap();

        assert_eq!(classifier.usage().cached_prompt_tokens, 3);
    }

    #[test]
    fn record_cached_prompt_tokens_returns_error_when_over_prompt_total() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(2);

        let result = classifier.record_cached_prompt_tokens(5);

        assert!(result.is_err());
    }

    #[test]
    fn markers_accessor_returns_configured_markers() {
        let configured = markers_with(Some(vec![token(1)]), Some(vec![token(2)]));
        let classifier = synthetic_classifier(configured);

        let returned = classifier.markers();

        assert_eq!(returned.reasoning_open.as_deref(), Some(&[token(1)][..]));
        assert_eq!(returned.reasoning_close.as_deref(), Some(&[token(2)][..]));
    }

    #[test]
    fn into_usage_consumes_classifier_and_yields_usage_snapshot() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(11);

        let usage = classifier.into_usage();

        assert_eq!(usage.prompt_tokens, 11);
    }

    #[test]
    fn spurious_tool_call_close_in_content_section_classifies_as_content() {
        // A `</tool_call>` while in Content (model misbehaves) is classified as
        // Content (not ToolCall) so observed_tool_calls isn't inflated.
        let mut markers = markers_with(None, None);
        markers.tool_call_close = Some(vec![token(300)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_pending(&mut classifier, 300, "</tool_call>");
        classifier.try_consume_marker_at_tail();
        let outcomes = classifier.drain_overflow();

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    fn markers_with_tool_call_open(tool_call_open: Vec<LlamaToken>) -> StreamingMarkers {
        StreamingMarkers {
            reasoning_open: None,
            reasoning_close: None,
            tool_call_open: Some(tool_call_open),
            tool_call_close: None,
        }
    }

    fn feed_json_string(
        classifier: &mut SampledTokenClassifier<'_>,
        text: &str,
        starting_token_id: i32,
    ) -> Vec<IngestOutcome> {
        let mut outcomes = Vec::new();
        for (offset, ch) in text.char_indices() {
            let token_id = starting_token_id + i32::try_from(offset).unwrap_or(i32::MAX);
            let mut buffer = [0_u8; 4];
            let chunk = ch.encode_utf8(&mut buffer);
            outcomes.extend(push_and_probe(classifier, token_id, chunk));
        }
        outcomes
    }

    #[test]
    fn json_probe_engages_when_first_non_whitespace_is_open_brace_in_content() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_and_probe(&mut classifier, 1, "{");

        assert!(matches!(classifier.probe_mode, ProbeMode::Active(_)));
    }

    #[test]
    fn json_probe_releases_tokens_as_tool_call_when_signature_matches() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"name":"f","arguments":{}}"#, 100);

        assert!(!outcomes.is_empty());
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
            "every emitted outcome should be ToolCall, got {:?}",
            outcome_sections(&outcomes),
        );
        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_signature_does_not_match() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"foo":"bar"}"#, 100);

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
            "every emitted outcome should be Content, got {:?}",
            outcome_sections(&outcomes),
        );
        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_extra_top_level_key() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{},"extra":1}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
        );
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_arguments_is_not_object() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"name":"f","arguments":"hi"}"#, 100);

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
        );
    }

    #[test]
    fn json_probe_handles_strings_with_quoted_braces_in_arguments() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"q":"a } b"}}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        );
    }

    #[test]
    fn json_probe_handles_escaped_quotes_in_string_values() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"q":"he said \"hi\""}}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        );
    }

    #[test]
    fn json_probe_handles_unicode_letters_in_strings() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"日本語","arguments":{"city":"パリ"}}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        );
    }

    #[test]
    fn json_probe_handles_nested_objects() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"a":{"b":{"c":1}}}}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        );
    }

    #[test]
    fn json_probe_handles_arrays_inside_arguments() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"items":[1,2,3]}}"#,
            100,
        );

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        );
    }

    #[test]
    fn json_probe_does_not_engage_when_first_byte_is_close_brace() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, "}}", 100);

        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
        );
    }

    #[test]
    fn json_probe_does_not_engage_in_reasoning_section() {
        let markers = StreamingMarkers {
            reasoning_open: Some(vec![token(800)]),
            reasoning_close: Some(vec![token(801)]),
            tool_call_open: Some(vec![token(900)]),
            tool_call_close: None,
        };
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_and_probe(&mut classifier, 1, "{");

        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
    }

    #[test]
    fn json_probe_does_not_engage_in_tool_call_section() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::ToolCall;

        push_and_probe(&mut classifier, 1, "{");

        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
    }

    #[test]
    fn marker_probe_takes_precedence_when_both_could_match() {
        // Marker is a single token whose decoded text starts with `"` (a JSON
        // signature-valid byte). The JSON probe holds the leading `{`, the
        // marker matches at the next token, the section transitions to ToolCall,
        // the JSON probe abandons. The leading `{` releases as Content; the
        // marker token releases as a ToolCall boundary (suppressed).
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        outcomes.extend(push_and_probe(&mut classifier, 1, "{"));
        outcomes.extend(push_and_probe(&mut classifier, 900, r#"""#));

        assert_eq!(classifier.section, SampledTokenSection::ToolCall);
        assert_eq!(outcome_pieces(&outcomes), vec!["{", ""]);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content, SampledTokenSection::ToolCall],
        );
    }

    #[test]
    fn json_probe_consumes_two_consecutive_objects_separately() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        outcomes.extend(feed_json_string(
            &mut classifier,
            r#"{"name":"a","arguments":{}}"#,
            100,
        ));
        outcomes.extend(feed_json_string(
            &mut classifier,
            r#"{"name":"b","arguments":{"x":1}}"#,
            200,
        ));

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
            "two consecutive markerless tool calls must both classify as ToolCall, got {:?}",
            outcome_sections(&outcomes),
        );
    }

    #[test]
    fn json_probe_with_leading_whitespace_then_open_brace_classifies_whitespace_as_content_and_json_as_tool_call()
     {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            "\n  {\"name\":\"f\",\"arguments\":{}}",
            100,
        );

        let tool_call_count = outcomes
            .iter()
            .filter(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_)))
            .count();
        let content_count = outcomes
            .iter()
            .filter(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_)))
            .count();
        assert_eq!(
            content_count, 3,
            "leading `\\n  ` should classify as content"
        );
        assert!(
            tool_call_count > 0,
            "the JSON object should classify as ToolCall",
        );
        assert_eq!(content_count + tool_call_count, outcomes.len());
    }

    #[test]
    fn json_probe_records_tool_call_token_usage_on_commit() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let json = r#"{"name":"f","arguments":{}}"#;
        let outcomes = feed_json_string(&mut classifier, json, 100);

        let emitted = outcomes.len();
        let usage = classifier.usage();
        assert_eq!(usage.tool_call_tokens, emitted as u64);
        assert_eq!(usage.content_tokens, 0);
    }

    #[test]
    fn json_probe_records_content_token_usage_on_abandon() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let json = r#"{"foo":"bar"}"#;
        let outcomes = feed_json_string(&mut classifier, json, 100);

        let emitted = outcomes.len();
        let usage = classifier.usage();
        assert_eq!(usage.content_tokens, emitted as u64);
        assert_eq!(usage.tool_call_tokens, 0);
    }

    #[test]
    fn flush_during_active_json_probe_releases_held_tokens_as_content() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_and_probe(&mut classifier, 1, "{");
        push_and_probe(&mut classifier, 2, r#""name""#);
        assert!(matches!(classifier.probe_mode, ProbeMode::Active(_)));

        let outcomes = classifier.flush();

        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
            "mid-probe flush must release held tokens as Content, got {:?}",
            outcome_sections(&outcomes),
        );
        assert!(matches!(classifier.probe_mode, ProbeMode::Idle));
    }
}
