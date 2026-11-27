//! Checks the extended batch, layer input, NextN, and state APIs on a tiny
//! CPU model that `LlamaModel::init_from_user` creates from metadata.

use std::num::NonZeroU32;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::{BatchExtAddError, LlamaBatchExt};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2 as sys;

const EMBEDDING_WIDTH: usize = 16;
const N_BATCH: u32 = 2;

/// The backend can be started only one time in a process, so the tests share it.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().unwrap())
}

fn create_model() -> LlamaModel {
    let mut metadata = GgufContext::new();
    metadata
        .set_val_str("general.architecture", "llama")
        .unwrap();
    metadata
        .set_val_str("tokenizer.ggml.model", "none")
        .unwrap();
    for (key, value) in [
        ("llama.vocab_size", 16),
        ("llama.context_length", 32),
        ("llama.embedding_length", EMBEDDING_WIDTH as u32),
        ("llama.block_count", 1),
        ("llama.feed_forward_length", 32),
        ("llama.attention.head_count", 1),
        ("llama.attention.head_count_kv", 1),
        ("llama.rope.dimension_count", EMBEDDING_WIDTH as u32),
    ] {
        metadata.set_val_u32(key, value).unwrap();
    }
    metadata
        .set_val_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .unwrap();

    let params = LlamaModelParams::default().with_n_gpu_layers(0);
    LlamaModel::init_from_user(
        backend(),
        &mut metadata,
        |tensor| unsafe {
            let size = sys::ggml_nbytes(tensor);
            let values = vec![0.01_f32; size / size_of::<f32>()];
            sys::ggml_backend_tensor_set(tensor, values.as_ptr().cast(), 0, size);
        },
        &params,
    )
    .unwrap()
}

fn create_context(model: &LlamaModel) -> LlamaContext<'_> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(32))
        .with_n_batch(N_BATCH)
        .with_n_ubatch(N_BATCH)
        .with_n_threads(1)
        .with_n_threads_batch(1)
        .with_flash_attention_policy(sys::LLAMA_FLASH_ATTN_TYPE_DISABLED);
    model.new_context(backend(), params).unwrap()
}

#[test]
fn reports_absent_dflash_selector_and_target_layers() {
    let model = create_model();
    assert_eq!(model.dflash_selector_top_k(), 0);
    assert!(model.target_layer_ids().is_empty());
}

#[test]
fn extended_batch_rejects_invalid_entries_and_recovers_capacity_after_clear() {
    let model = create_model();
    let mut context = create_context(&model);
    let mut batch = LlamaBatchExt::new(&mut context);
    assert_eq!(batch.add_embedding_row(-1, 0), Err(BatchExtAddError(-3)));
    assert_eq!(batch.add_embedding_row(0, -1), Err(BatchExtAddError(-2)));
    batch.clear();
    assert_eq!(batch.add_embedding_row(0, 0), Ok(0));
    assert_eq!(batch.add_embedding_row(0, 1), Ok(1));
    assert_eq!(batch.add_embedding_row(0, 2), Err(BatchExtAddError(-1)));
    batch.clear();
    assert_eq!(batch.add_embedding_row(0, 0), Ok(0));
}

#[test]
fn extended_batch_validates_state_rows_and_entry_configuration() {
    let model = create_model();
    let mut context = create_context(&model);
    let mut batch = LlamaBatchExt::new(&mut context);
    let index = batch.add_token(0, LlamaToken(0)).unwrap();
    assert!(batch.set_embedding_state_row(index, -1).is_err());
    assert!(batch.set_embedding_state_row(index + 1, 0).is_err());
    assert!(batch.set_embedding_state_row(index, 0).is_ok());
    assert!(batch.set_embedding_state_row(index, 1).is_err());
    assert!(batch.set_positions(index, [0; 4]).is_ok());
    assert!(batch.set_positions(index + 1, [0; 4]).is_err());
    assert!(batch.set_output_logits(index, true).is_ok());
    assert!(batch.set_output_embeddings(index + 1, true).is_err());
    assert!(batch.add_sequence_id(index, -1).is_err());
}

#[test]
fn extended_decode_exposes_layer_inputs_and_state_copy_respects_slice_bounds() {
    let model = create_model();
    let mut context = create_context(&model);
    context.set_embeddings_layer_inp(0, true);
    let mut batch = LlamaBatchExt::new(&mut context);
    let index = batch.add_token(0, LlamaToken(0)).unwrap();
    batch.set_positions(index, [0; 4]).unwrap();
    batch.set_output_embeddings(index, true).unwrap();
    batch.decode().unwrap();

    // One row for each batch slot; the decoded token is in the first row.
    let embeddings = context.embeddings_layer_inp(0);
    assert_eq!(embeddings.len(), EMBEDDING_WIDTH * N_BATCH as usize);
    assert_eq!(&embeddings[..EMBEDDING_WIDTH], &[0.01_f32; EMBEDDING_WIDTH]);

    let size = context.get_state_size();
    assert!(size > 0);
    let mut guarded = vec![0xa5; size + 8];
    assert_eq!(context.copy_state_data(&mut guarded[..size]), size);
    assert_eq!(&guarded[size..], &[0xa5; 8]);
}

#[test]
fn state_copy_rejects_a_short_slice_without_overwriting_its_guard() {
    let model = create_model();
    let context = create_context(&model);
    let size = context.get_state_size();
    let mut guarded = vec![0xa5; size + 8];
    assert_eq!(context.copy_state_data(&mut guarded[..size - 1]), 0);
    assert_eq!(&guarded[size - 1..], &[0xa5; 9]);
}

#[test]
fn nextn_transfers_reject_unconfigured_buffers() {
    let model = create_model();
    let mut context = create_context(&model);
    let mut output = [0.5; EMBEDDING_WIDTH + 1];
    assert!(context.embeddings_nextn_rows(0, &mut output).is_err());
    assert!(context.set_embeddings_nextn_rows(0, &output).is_err());
    assert!(context.set_embeddings_nextn_dev(true).is_err());
    context.set_embeddings_nextn(true, true);
    assert!(context.set_embeddings_nextn_dev(true).is_err());
    context.set_embeddings_nextn(true, false);

    // The CPU backend has no events, so even unmasked extraction cannot use a
    // device buffer, and a rejected copy leaves the host buffer unchanged.
    assert!(context.set_embeddings_nextn_dev(true).is_err());
    assert!(context.embeddings_nextn_rows(0, &mut output).is_err());
    assert_eq!(output, [0.5; EMBEDDING_WIDTH + 1]);
    assert!(context.set_embeddings_nextn_dev(false).is_ok());
}

#[test]
fn reports_unpaired_context_and_rejects_unsupported_state_source() {
    let model = create_model();
    let mut source = create_context(&model);
    assert!(source.ctx_other_ptr().is_null());
    let mut destination = create_context(&model);
    assert!(destination.ctx_other_ptr().is_null());
    assert!(!destination.supports_embedding_rows());

    // A rejected source is still kept, so clear it while the source is alive.
    assert!(unsafe { destination.set_context_state_source(&mut source) }.is_err());
    destination.clear_context_state_source();
}
