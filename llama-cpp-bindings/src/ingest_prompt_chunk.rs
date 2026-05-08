use crate::mtmd::MtmdInputChunk;
use crate::mtmd::MtmdInputChunkType;
use crate::mtmd::MtmdInputChunkTypeError;
use crate::sampled_token_classifier::SampledTokenClassifier;

/// Dispatches a single multimodal chunk into the classifier:
/// - Text chunks bump `prompt_tokens` and replay every text token through the
///   marker state machine, so prompt-end markers like `<think>` reach the
///   classifier and the section transitions before generation begins.
/// - Image / Audio chunks bump only their own usage counters; they have no
///   text token IDs to replay.
///
/// This is the single canonical per-chunk ingest path for the multimodal
/// driver. Any future per-chunk invariant (e.g. cached prefix replay) lives
/// here so it cannot diverge between consumers.
///
/// # Errors
/// Returns [`MtmdInputChunkTypeError`] when the chunk reports a type unknown
/// to this binding. Counters are not updated on error.
pub fn ingest_prompt_chunk(
    classifier: &mut SampledTokenClassifier<'_>,
    chunk: &MtmdInputChunk,
) -> Result<(), MtmdInputChunkTypeError> {
    let n_tokens = chunk.n_tokens() as u64;
    match chunk.chunk_type()? {
        MtmdInputChunkType::Text => {
            classifier.record_prompt_tokens(n_tokens);
            if let Some(tokens) = chunk.text_tokens() {
                classifier.ingest_prompt_tokens(tokens);
            }
        }
        MtmdInputChunkType::Image => classifier.record_input_image_tokens(n_tokens),
        MtmdInputChunkType::Audio => classifier.record_input_audio_tokens(n_tokens),
    }

    Ok(())
}
