use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use llama_cpp_bindings::DecodeError;
use llama_cpp_bindings::LogitsError;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaLoraAdapter;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::TestFixture;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::test_model;
use serial_test::serial;

#[test]
#[serial]
fn context_creation_and_properties() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    assert!(context.n_ctx() > 0);
    assert!(context.n_batch() > 0);
    assert!(context.n_ubatch() > 0);

    Ok(())
}

#[test]
#[serial]
fn decode_and_get_logits() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let decode_result = context.decode(&mut batch);
    assert!(decode_result.is_ok());

    let logits = context.get_logits()?;
    assert!(!logits.is_empty());

    Ok(())
}

#[test]
#[serial]
fn timings_work() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;

    context.reset_timings();
    let timings = context.timings();
    assert!(timings.t_start_ms() >= 0.0);

    Ok(())
}

#[test]
#[serial]
fn token_data_array_has_entries_after_decode() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let token_data_array = context.token_data_array()?;

    assert!(!token_data_array.data.is_empty());

    Ok(())
}

#[test]
#[serial]
fn get_logits_ith_returns_valid_slice() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let logits = context.get_logits_ith(last_index)?;

    assert_eq!(logits.len(), usize::try_from(model.n_vocab())?);

    Ok(())
}

#[test]
#[serial]
fn token_data_array_ith_returns_valid_data() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let token_data_array = context.token_data_array_ith(last_index)?;

    assert_eq!(
        token_data_array.data.len(),
        usize::try_from(model.n_vocab())?
    );

    Ok(())
}

#[test]
#[serial]
fn embeddings_ith_returns_error_when_embeddings_disabled() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(false);
    let context = model.new_context(backend, ctx_params)?;

    let result = context.embeddings_ith(0);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn embeddings_seq_ith_returns_error_when_embeddings_disabled() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(false);
    let context = model.new_context(backend, ctx_params)?;

    let result = context.embeddings_seq_ith(0);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn candidates_returns_n_vocab_entries() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let count = context.candidates()?.count();

    assert_eq!(count, usize::try_from(model.n_vocab())?);

    Ok(())
}

#[test]
#[serial]
fn debug_format_contains_struct_name() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;
    let debug_output = format!("{context:?}");

    assert!(debug_output.contains("LlamaContext"));

    Ok(())
}

#[test]
#[serial]
fn decode_with_embeddings_enabled() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn embeddings_seq_ith_returns_valid_embeddings() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let embeddings = context.embeddings_seq_ith(0)?;

    assert_eq!(embeddings.len(), usize::try_from(model.n_embd())?);

    Ok(())
}

#[test]
#[serial]
fn multi_sequence_embeddings_returns_one_embedding_per_sequence() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_n_seq_max(4)
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;

    let inputs = [
        "alpha is here",
        "beta runs fast",
        "gamma waits",
        "delta jumps",
    ];
    let mut batch = LlamaBatch::new(64, 4)?;

    for (sequence_index, text) in inputs.iter().enumerate() {
        let tokens = model.str_to_token(text, AddBos::Always)?;
        let sequence_id = i32::try_from(sequence_index)?;

        batch.add_sequence(&tokens, sequence_id, true)?;
    }

    context.decode(&mut batch)?;

    let n_embd = usize::try_from(model.n_embd())?;
    let mut collected: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());

    for sequence_index in 0..inputs.len() {
        let sequence_id = i32::try_from(sequence_index)?;
        let embedding = context.embeddings_seq_ith(sequence_id)?;

        assert_eq!(
            embedding.len(),
            n_embd,
            "sequence {sequence_index} embedding length mismatch"
        );

        collected.push(embedding.to_vec());
    }

    for (left_index, left) in collected.iter().enumerate() {
        for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
            assert_ne!(
                left, right,
                "embedding for sequence {left_index} must differ from sequence {right_index}",
            );
        }
    }

    Ok(())
}

/// Reproduces paddler's embedding batching loop exactly with the document strings, batch
/// shape, and iteration pattern from the failing harness test
/// `agent_embedding_batch_distribution_independent_of_context_size`. A `LlamaBatch` is
/// allocated once with `n_tokens=64` and `n_seq_max=4`, then reused across two iterations
/// of two sequences each (because the four ~22-token docs do not all fit in one
/// 64-token window). Per iteration: `add_sequence` for each doc, `clear_kv_cache`,
/// `decode`, `embeddings_seq_ith` for each filled slot, `batch.clear()`. Every iteration
/// must yield distinct, non-empty embeddings — including iterations after the first.
#[test]
#[serial]
fn embeddings_returns_distinct_values_when_reused_batch_has_extra_capacity() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_n_seq_max(4)
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;

    let iterations = [
        [
            "This is the first document with enough content to contribute meaningfully to the batch size calculation",
            "This is the second document that should be processed in a potentially different batch from the first",
        ],
        [
            "This is the third document adding more content to ensure the total exceeds the configured chunk limit",
            "This is the fourth document which should demonstrate that batching distributes across agent requests",
        ],
    ];

    let n_embd = usize::try_from(model.n_embd())?;
    let mut batch = LlamaBatch::new(64, 4)?;
    let mut collected: Vec<Vec<f32>> = Vec::new();

    for iteration_inputs in iterations {
        for (sequence_index, text) in iteration_inputs.iter().enumerate() {
            let tokens = model.str_to_token(text, AddBos::Always)?;
            let sequence_id = i32::try_from(sequence_index)?;

            batch.add_sequence(&tokens, sequence_id, true)?;
        }

        context.clear_kv_cache();
        context.decode(&mut batch)?;

        for sequence_index in 0..iteration_inputs.len() {
            let sequence_id = i32::try_from(sequence_index)?;
            let embedding = context.embeddings_seq_ith(sequence_id)?;

            assert_eq!(
                embedding.len(),
                n_embd,
                "iteration sequence {sequence_index} embedding length mismatch"
            );

            collected.push(embedding.to_vec());
        }

        batch.clear();
    }

    assert_eq!(
        collected.len(),
        iterations.iter().flatten().count(),
        "expected one embedding per input across every iteration"
    );

    for (left_index, left) in collected.iter().enumerate() {
        for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
            assert_ne!(
                left, right,
                "embedding {left_index} must differ from embedding {right_index} across reused-batch iterations",
            );
        }
    }

    Ok(())
}

#[test]
#[serial]
fn embeddings_ith_returns_valid_embeddings() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let embeddings = context.embeddings_ith(last_index)?;

    assert_eq!(embeddings.len(), usize::try_from(model.n_embd())?);

    Ok(())
}

#[test]
#[serial]
fn candidates_ith_returns_n_vocab_entries() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let count = context.candidates_ith(last_index)?.count();

    assert_eq!(count, usize::try_from(model.n_vocab())?);

    Ok(())
}

#[test]
#[serial]
fn lora_adapter_remove_succeeds_with_no_adapters() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;
    let mut adapter = LlamaLoraAdapter {
        lora_adapter: NonNull::dangling(),
    };

    let result = context.lora_adapter_remove(&mut adapter);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn encode_on_non_encoder_model_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&mut batch);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn lora_adapter_set_with_dangling_pointer_succeeds_or_errors() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;
    let mut adapter = LlamaLoraAdapter {
        lora_adapter: NonNull::dangling(),
    };

    let result = context.lora_adapter_set(&mut adapter, 1.0);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn embeddings_ith_returns_null_embedding_error_for_non_embedding_token() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.embedding_model()?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let context = model.new_context(backend, ctx_params)?;

    let result = context.embeddings_ith(999);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn embeddings_seq_ith_returns_null_embedding_error_for_invalid_seq() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.embeddings_seq_ith(999);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn decode_empty_batch_returns_error() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new(512, 1)?;

    let result = context.decode(&mut batch);

    assert!(result.is_err());

    Ok(())
}

#[test]
#[serial]
fn encode_succeeds_with_encoder_model() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model_path = test_model::download_encoder_model()?;
    let model_params = inference_model_params();
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(512))
        .with_embeddings(true);
    let mut context = model.new_context(backend, ctx_params)?;
    let tokens = model.str_to_token("hello", AddBos::Never)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn set_abort_flag_aborts_decode() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let abort_flag = Arc::new(AtomicBool::new(true));
    context.set_abort_flag(abort_flag);

    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert_eq!(result, Err(DecodeError::Aborted));

    Ok(())
}

#[test]
#[serial]
fn set_abort_flag_false_allows_decode() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let abort_flag = Arc::new(AtomicBool::new(false));
    context.set_abort_flag(abort_flag);

    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn clear_abort_callback_allows_decode_with_flag_true() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(backend, ctx_params)?;
    let abort_flag = Arc::new(AtomicBool::new(true));
    context.set_abort_flag(abort_flag);
    context.clear_abort_callback();

    let tokens = model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[test]
#[serial]
fn synchronize_completes_without_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    context.synchronize();

    Ok(())
}

#[test]
#[serial]
fn detach_threadpool_completes_without_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    context.detach_threadpool();

    Ok(())
}

#[test]
#[serial]
fn get_logits_ith_returns_token_not_initialized_for_unknown_index() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let context = model.new_context(backend, ctx_params)?;

    let result = context.get_logits_ith(7);

    assert!(matches!(result, Err(LogitsError::TokenNotInitialized(7))));

    Ok(())
}

#[test]
#[serial]
fn get_logits_ith_returns_token_index_exceeds_context_for_huge_index() -> Result<()> {
    let fixture = TestFixture::shared();
    let backend = fixture.backend();
    let model = fixture.default_model();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(64));
    let mut context = model.new_context(backend, ctx_params)?;

    let huge_index = i32::try_from(context.n_ctx())?;
    context.mark_logits_initialized(huge_index);
    let result = context.get_logits_ith(huge_index);

    assert!(matches!(
        result,
        Err(LogitsError::TokenIndexExceedsContext { .. })
    ));

    Ok(())
}
