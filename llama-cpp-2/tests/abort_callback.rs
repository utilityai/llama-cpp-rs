//! Set `LLAMA_TEST_EMBEDDING_GGUF` to a GGUF embedding model to exercise native
//! CPU cancellation. No model is downloaded by this test.

use llama_cpp_2::context::params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::DecodeError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[test]
fn callback_controls_cpu_decode() {
    let Some(path) = std::env::var_os("LLAMA_TEST_EMBEDDING_GGUF") else {
        return;
    };
    let backend = LlamaBackend::init().unwrap();
    // Context creation initializes selected GPU backends even with zero
    // offloaded layers. Exclude them explicitly to keep this test CPU-only.
    let model = LlamaModel::load_from_file(
        &backend,
        path,
        &LlamaModelParams::default()
            .with_n_gpu_layers(0)
            .with_devices(&[])
            .unwrap(),
    )
    .unwrap();
    let params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Cls)
        .with_attention_type(LlamaAttentionType::NonCausal)
        .with_offload_kqv(false)
        .with_op_offload(false);
    let mut context = model.new_context(&backend, params).unwrap();
    let tokens = model.vocab().tokenize(b"Test cancellation.", true, true);
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch.add_sequence(&tokens, 0, true).unwrap();

    let cancel = Arc::new(AtomicBool::new(true));
    context.set_abort_callback({
        let cancel = Arc::clone(&cancel);
        move || cancel.load(Ordering::Relaxed)
    });
    assert_eq!(context.decode(&mut batch), Err(DecodeError::Aborted));

    cancel.store(false, Ordering::Relaxed);
    context.clear_kv_cache();
    context.decode(&mut batch).unwrap();
    let expected = context.embeddings_seq_ith(0).unwrap().to_vec();

    context.set_abort_callback(|| true);
    context.clear_kv_cache();
    assert_eq!(context.decode(&mut batch), Err(DecodeError::Aborted));
    context.clear_abort_callback();
    context.clear_abort_callback();
    context.clear_kv_cache();
    context.decode(&mut batch).unwrap();
    assert_eq!(context.embeddings_seq_ith(0).unwrap(), expected);
}
