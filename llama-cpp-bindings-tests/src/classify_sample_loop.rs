use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;

/// Drives a classifier through the full sample/decode/flush loop.
///
/// Suppresses EOG outcomes (so `generated_raw` and the per-section streams
/// never contain end-of-generation marker text) and captures per-section
/// counts. Tests that need to exercise classifier behaviour during real
/// inference should construct one of these and call
/// [`ClassifySampleLoop::run`] instead of re-implementing the loop. The
/// strict per-test assertions then run on [`ClassifySampleLoopOutcome`].
pub struct ClassifySampleLoop<'borrow, 'model, 'tokens> {
    pub model: &'model LlamaModel,
    pub classifier: &'borrow mut SampledTokenClassifier<'model>,
    pub sampler: &'borrow mut LlamaSampler,
    pub context: &'borrow mut LlamaContext<'model>,
    pub batch: &'borrow mut LlamaBatch<'tokens>,
    pub initial_position: i32,
    pub max_generated_tokens: i32,
}

#[derive(Debug, Default)]
pub struct ClassifySampleLoopOutcome {
    pub generated_raw: String,
    pub content_stream: String,
    pub reasoning_stream: String,
    pub observed_content: u64,
    pub observed_reasoning: u64,
    pub observed_tool_call: u64,
    pub observed_undeterminable: u64,
    pub eog_seen: bool,
}

impl ClassifySampleLoop<'_, '_, '_> {
    /// # Errors
    /// Forwards [`SampledTokenClassifier::sample`] / [`LlamaContext::decode`] /
    /// [`LlamaBatch::add`] errors verbatim. Stops on EOG, on
    /// `max_generated_tokens` exhaustion, or on the first error.
    pub fn run(self) -> Result<ClassifySampleLoopOutcome> {
        let mut outcome = ClassifySampleLoopOutcome::default();
        let mut position = self.initial_position;
        let max_position = position + self.max_generated_tokens;

        while position < max_position {
            let (raw_token, ingest_outcomes) =
                self.classifier
                    .sample(self.sampler, self.context, self.batch.n_tokens() - 1)?;

            for ingest_outcome in &ingest_outcomes {
                let is_eog = self.model.is_eog_token(&ingest_outcome.sampled_token);
                if is_eog {
                    outcome.eog_seen = true;
                } else {
                    outcome.generated_raw.push_str(&ingest_outcome.raw_piece);
                }
                // Counters always include EOG so they match the classifier's
                // internal usage counters (which include every sampled token).
                // EOG text is suppressed from `generated_raw` and the per-section
                // streams so callers can assert exact textual equality.
                record_outcome(ingest_outcome, &mut outcome, is_eog);
            }

            let raw_as_sampled = SampledToken::Content(raw_token);
            if self.model.is_eog_token(&raw_as_sampled) {
                outcome.eog_seen = true;
                break;
            }

            self.batch.clear();
            self.batch.add(&raw_as_sampled, position, &[0], true)?;
            position += 1;

            self.context.decode(self.batch)?;
        }

        for ingest_outcome in self.classifier.flush() {
            let is_eog = self.model.is_eog_token(&ingest_outcome.sampled_token);
            if is_eog {
                outcome.eog_seen = true;
            } else {
                outcome.generated_raw.push_str(&ingest_outcome.raw_piece);
            }
            record_outcome(&ingest_outcome, &mut outcome, is_eog);
        }

        Ok(outcome)
    }
}

fn record_outcome(ingest: &IngestOutcome, outcome: &mut ClassifySampleLoopOutcome, is_eog: bool) {
    match ingest.sampled_token {
        SampledToken::Content(_) => {
            outcome.observed_content += 1;
            if !is_eog {
                outcome.content_stream.push_str(&ingest.visible_piece);
            }
        }
        SampledToken::Reasoning(_) => {
            outcome.observed_reasoning += 1;
            if !is_eog {
                outcome.reasoning_stream.push_str(&ingest.visible_piece);
            }
        }
        SampledToken::ToolCall(_) => {
            outcome.observed_tool_call += 1;
        }
        SampledToken::Undeterminable(_) => {
            outcome.observed_undeterminable += 1;
        }
    }
}
