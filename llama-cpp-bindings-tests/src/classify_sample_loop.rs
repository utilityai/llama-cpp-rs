use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;

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

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::ingest_outcome::IngestOutcome;
    use llama_cpp_bindings::sampled_token::SampledToken;
    use llama_cpp_bindings::token::LlamaToken;

    use super::ClassifySampleLoopOutcome;
    use super::record_outcome;

    #[test]
    fn record_outcome_tool_call_token() {
        let ingest = IngestOutcome {
            sampled_token: SampledToken::ToolCall(LlamaToken(42)),
            visible_piece: String::new(),
            raw_piece: String::new(),
        };
        let mut outcome = ClassifySampleLoopOutcome::default();

        record_outcome(&ingest, &mut outcome, false);

        assert_eq!(outcome.observed_tool_call, 1);
        assert_eq!(outcome.observed_content, 0);
        assert_eq!(outcome.observed_reasoning, 0);
        assert_eq!(outcome.observed_undeterminable, 0);
    }

    #[test]
    fn record_outcome_reasoning_token_streams_visible_piece() {
        let ingest = IngestOutcome {
            sampled_token: SampledToken::Reasoning(LlamaToken(7)),
            visible_piece: "thinking".to_string(),
            raw_piece: String::new(),
        };
        let mut outcome = ClassifySampleLoopOutcome::default();

        record_outcome(&ingest, &mut outcome, false);

        assert_eq!(outcome.observed_reasoning, 1);
        assert_eq!(outcome.reasoning_stream, "thinking");
    }

    #[test]
    fn record_outcome_reasoning_token_at_end_of_generation_is_not_streamed() {
        let ingest = IngestOutcome {
            sampled_token: SampledToken::Reasoning(LlamaToken(7)),
            visible_piece: "thinking".to_string(),
            raw_piece: String::new(),
        };
        let mut outcome = ClassifySampleLoopOutcome::default();

        record_outcome(&ingest, &mut outcome, true);

        assert_eq!(outcome.observed_reasoning, 1);
        assert!(outcome.reasoning_stream.is_empty());
    }

    #[test]
    fn record_outcome_undeterminable_token_counts_without_streaming() {
        let ingest = IngestOutcome {
            sampled_token: SampledToken::Undeterminable(LlamaToken(9)),
            visible_piece: "ignored".to_string(),
            raw_piece: String::new(),
        };
        let mut outcome = ClassifySampleLoopOutcome::default();

        record_outcome(&ingest, &mut outcome, false);

        assert_eq!(outcome.observed_undeterminable, 1);
        assert!(outcome.content_stream.is_empty());
        assert!(outcome.reasoning_stream.is_empty());
    }
}
