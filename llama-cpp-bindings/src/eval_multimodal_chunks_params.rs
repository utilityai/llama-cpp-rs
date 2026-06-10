use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_sys::llama_seq_id;

/// Settings for one `eval_multimodal_chunks` call on a `SampledTokenClassifier`.
#[derive(Clone, Copy, Debug)]
pub struct EvalMultimodalChunksParams {
    /// Position of the first chunk token within the target sequence.
    pub start_position: llama_pos,
    /// Sequence id under which the chunks are evaluated.
    pub seq_id: llama_seq_id,
    /// Logical batch size for splitting chunk tokens into decode batches.
    pub n_batch: i32,
    /// Whether logits are requested for the final token of the final chunk.
    pub logits_last: bool,
}
