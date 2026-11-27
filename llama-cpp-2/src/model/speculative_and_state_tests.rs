// This in-memory fixture needs the crate-private LlamaModel field and Ptr type,
// so these tests live inside the model module rather than in tests/. This avoids
// exposing test-only constructors or requiring external model files.

use super::*;
use crate::llama_batch::{BatchExtAddError, LlamaBatchExt};
use llama_cpp_sys_2 as sys;
use std::num::NonZeroU32;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

const EMBEDDING_WIDTH: usize = 16;

struct ApiFixture {
    model: LlamaModel,
    backend: LlamaBackend,
}

impl ApiFixture {
    fn new() -> Self {
        let backend = LlamaBackend::init().expect("initialize test backend");

        // The upstream virtual-model loader creates its tensors from metadata, so
        // these tests need neither downloaded weights nor a generated model file.
        unsafe extern "C" fn initialize_tensor(
            tensor: *mut sys::ggml_tensor,
            _: *mut std::ffi::c_void,
        ) {
            let size = unsafe { sys::ggml_nbytes(tensor) };
            let values = vec![0.01_f32; size / std::mem::size_of::<f32>()];
            unsafe { sys::ggml_backend_tensor_set(tensor, values.as_ptr().cast(), 0, size) };
        }

        let model = unsafe {
            let metadata = sys::gguf_init_empty();
            assert!(!metadata.is_null());
            sys::gguf_set_val_str(
                metadata,
                c"general.architecture".as_ptr(),
                c"llama".as_ptr(),
            );
            sys::gguf_set_val_str(metadata, c"tokenizer.ggml.model".as_ptr(), c"none".as_ptr());
            for (key, value) in [
                (c"llama.vocab_size", 16),
                (c"llama.context_length", 32),
                (c"llama.embedding_length", EMBEDDING_WIDTH as u32),
                (c"llama.block_count", 1),
                (c"llama.feed_forward_length", 32),
                (c"llama.attention.head_count", 1),
                (c"llama.attention.head_count_kv", 1),
                (c"llama.rope.dimension_count", EMBEDDING_WIDTH as u32),
            ] {
                sys::gguf_set_val_u32(metadata, key.as_ptr(), value);
            }
            sys::gguf_set_val_f32(
                metadata,
                c"llama.attention.layer_norm_rms_epsilon".as_ptr(),
                1e-5,
            );
            let mut params = sys::llama_model_default_params();
            params.n_gpu_layers = 0;
            let model = sys::llama_model_init_from_user(
                metadata,
                Some(initialize_tensor),
                std::ptr::null_mut(),
                params,
            );
            sys::gguf_free(metadata);
            LlamaModel {
                model: Ptr::new(model).expect("create tiny CPU model"),
            }
        };
        Self { model, backend }
    }

    fn create_context(&self) -> LlamaContext<'_> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(32))
            .with_n_batch(2)
            .with_n_ubatch(2)
            .with_n_threads(1)
            .with_n_threads_batch(1)
            .with_flash_attention_policy(sys::LLAMA_FLASH_ATTN_TYPE_DISABLED);
        self.model
            .new_context(&self.backend, params)
            .expect("create test context")
    }

    fn run(check: impl FnOnce(&Self, LlamaContext<'_>)) {
        static FIXTURE: OnceLock<Mutex<ApiFixture>> = OnceLock::new();

        // Backend initialization is process-wide, and native context creation can
        // mutate the model: serialize these tests rather than sharing it concurrently.
        let fixture = FIXTURE
            .get_or_init(|| Mutex::new(Self::new()))
            .lock()
            .unwrap();
        check(&fixture, fixture.create_context());
    }
}

#[test]
fn reports_absent_dflash_selector_and_target_layers() {
    ApiFixture::run(|fixture, _| {
        assert_eq!(fixture.model.get_dflash_selector_top_k(), 0);
        assert!(fixture.model.get_target_layer_ids().is_empty());
    });
}

#[test]
fn extended_batch_rejects_invalid_entries_and_recovers_capacity_after_clear() {
    ApiFixture::run(|_, mut context| {
        let mut batch = LlamaBatchExt::new(&mut context).unwrap();
        assert_eq!(batch.add_embedding_row(-1, 0), Err(BatchExtAddError(-3)));
        assert_eq!(batch.add_embedding_row(0, -1), Err(BatchExtAddError(-2)));
        batch.clear();
        assert_eq!(batch.add_embedding_row(0, 0), Ok(0));
        assert_eq!(batch.add_embedding_row(0, 1), Ok(1));
        assert_eq!(batch.add_embedding_row(0, 2), Err(BatchExtAddError(-1)));
        batch.clear();
        assert_eq!(batch.add_embedding_row(0, 0), Ok(0));
    });
}

#[test]
fn extended_batch_validates_state_rows_and_entry_configuration() {
    ApiFixture::run(|_, mut context| {
        let mut batch = LlamaBatchExt::new(&mut context).unwrap();
        let index = batch.add_token(0, LlamaToken(0)).unwrap();
        assert!(!batch.set_embedding_state_row(index, -1));
        assert!(!batch.set_embedding_state_row(index + 1, 0));
        assert!(batch.set_embedding_state_row(index, 0));
        assert!(!batch.set_embedding_state_row(index, 1));
        assert!(batch.set_positions(index, [0; 4]));
        assert!(!batch.set_positions(index + 1, [0; 4]));
        assert!(batch.set_output_logits(index, true));
        assert!(!batch.set_output_embeddings(index + 1, true));
        assert!(!batch.add_sequence_id(index, -1));
    });
}

#[test]
fn extended_decode_exposes_layer_inputs_and_state_copy_respects_slice_bounds() {
    ApiFixture::run(|_, mut context| {
        context.set_embeddings_layer_inp(0, true);
        let mut batch = LlamaBatchExt::new(&mut context).unwrap();
        let index = batch.add_token(0, LlamaToken(0)).unwrap();
        assert!(batch.set_positions(index, [0; 4]));
        assert!(batch.set_output_embeddings(index, true));
        batch.decode().unwrap();

        // One decoded token initializes exactly one layer-input row in this fixture.
        let embeddings = unsafe { context.get_embeddings_layer_inp(0, EMBEDDING_WIDTH) };
        assert_eq!(embeddings, &[0.01_f32; EMBEDDING_WIDTH]);
        let size = context.get_state_size();
        assert!(size > 0);
        let mut guarded = vec![0xa5; size + 8];
        assert_eq!(context.copy_state_data(&mut guarded[..size]), size);
        assert_eq!(&guarded[size..], &[0xa5; 8]);
    });
}

#[test]
fn state_copy_rejects_a_short_slice_without_overwriting_its_guard() {
    ApiFixture::run(|_, context| {
        let size = context.get_state_size();
        let mut guarded = vec![0xa5; size + 8];
        assert_eq!(context.copy_state_data(&mut guarded[..size - 1]), 0);
        assert_eq!(&guarded[size - 1..], &[0xa5; 9]);
    });
}

#[test]
fn nextn_transfers_reject_partial_rows_and_unconfigured_buffers() {
    ApiFixture::run(|_, mut context| {
        let mut output = [0.5; EMBEDDING_WIDTH];
        assert!(!context.get_embeddings_nextn_rows(0, &mut output));
        assert_eq!(output, [0.5; EMBEDDING_WIDTH]);
        assert!(!context.set_embeddings_nextn_rows(0, &output));
        assert!(!context.set_embeddings_nextn_dev(true));
        context.set_embeddings_nextn(true, true);
        assert!(!context.set_embeddings_nextn_dev(true));
        context.set_embeddings_nextn(true, false);

        // CPU has no backend events, so even unmasked extraction cannot enable
        // a device buffer; rejection must leave host destination memory untouched.
        assert!(!context.set_embeddings_nextn_dev(true));
        assert!(!context.get_embeddings_nextn_rows(0, &mut output));
        assert_eq!(output, [0.5; EMBEDDING_WIDTH]);
        assert!(context.set_embeddings_nextn_dev(false));

        // Keep the fixture lock outside each unwind so an expected bounds panic
        // does not poison the shared model and obscure subsequent test failures.
        assert!(catch_unwind(AssertUnwindSafe(|| {
            context.get_embeddings_nextn_rows(0, &mut output[..EMBEDDING_WIDTH - 1]);
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            context.set_embeddings_nextn_rows(0, &output[..EMBEDDING_WIDTH - 1]);
        }))
        .is_err());
    });
}

#[test]
fn reports_unpaired_context_and_rejects_unsupported_state_source() {
    ApiFixture::run(|fixture, mut source| {
        assert!(source.get_context_other_ptr().is_null());
        let mut destination = fixture.create_context();
        assert!(destination.get_context_other_ptr().is_null());
        assert!(!destination.supports_embedding_rows());

        // Even a rejected state source is retained natively; clear it while the
        // source is still alive, and destroy the destination before the source.
        unsafe {
            assert!(!destination.set_context_state_source(Some(&mut source)));
            assert!(!destination.set_context_state_source(None));
        }
    });
}
