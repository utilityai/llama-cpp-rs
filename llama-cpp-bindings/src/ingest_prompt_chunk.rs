use crate::mtmd::MtmdInputChunk;
use crate::mtmd::MtmdInputChunkType;
use crate::mtmd::MtmdInputChunkTypeError;
use crate::sampled_token_classifier::SampledTokenClassifier;

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
