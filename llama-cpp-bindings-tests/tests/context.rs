use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use llama_cpp_bindings::DecodeError;
use llama_cpp_bindings::LogitsError;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaLoraAdapter;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;

// =========================================================================================
// Group A: default Qwen model, embeddings=false. Most context tests fall here.
// =========================================================================================

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn context_creation_and_properties(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert!(context.n_ctx() > 0);
    assert!(context.n_batch() > 0);
    assert!(context.n_ubatch() > 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decode_and_get_logits(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let decode_result = context.decode(&mut batch);
    assert!(decode_result.is_ok());

    let logits = context.get_logits()?;
    assert!(!logits.is_empty());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn timings_work(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.reset_timings();
    let timings = context.timings();
    assert!(timings.t_start_ms() >= 0.0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn token_data_array_has_entries_after_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let token_data_array = context.token_data_array()?;

    assert!(!token_data_array.data.is_empty());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn get_logits_ith_returns_valid_slice(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let logits = context.get_logits_ith(last_index)?;

    assert_eq!(logits.len(), usize::try_from(fixture.model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn token_data_array_ith_returns_valid_data(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let token_data_array = context.token_data_array_ith(last_index)?;

    assert_eq!(
        token_data_array.data.len(),
        usize::try_from(fixture.model.n_vocab())?
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn embeddings_ith_returns_error_when_embeddings_disabled(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let result = context.embeddings_ith(0);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn embeddings_seq_ith_returns_error_when_embeddings_disabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let result = context.embeddings_seq_ith(0);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn candidates_returns_n_vocab_entries(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let count = context.candidates()?.count();

    assert_eq!(count, usize::try_from(fixture.model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn debug_format_contains_struct_name(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let debug_output = format!("{context:?}");

    assert!(debug_output.contains("LlamaContext"));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn candidates_ith_returns_n_vocab_entries(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let count = context.candidates_ith(last_index)?.count();

    assert_eq!(count, usize::try_from(fixture.model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn lora_adapter_remove_succeeds_with_no_adapters(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut adapter = LlamaLoraAdapter {
        lora_adapter: NonNull::dangling(),
    };

    let result = context.lora_adapter_remove(&mut adapter);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn encode_on_non_encoder_model_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&mut batch);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn lora_adapter_set_with_dangling_pointer_succeeds_or_errors(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut adapter = LlamaLoraAdapter {
        lora_adapter: NonNull::dangling(),
    };

    let result = context.lora_adapter_set(&mut adapter, 1.0);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_seq_ith_returns_null_embedding_error_for_invalid_seq(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let result = context.embeddings_seq_ith(999);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decode_empty_batch_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut batch = LlamaBatch::new(512, 1)?;

    let result = context.decode(&mut batch);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn set_abort_flag_aborts_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let abort_flag = Arc::new(AtomicBool::new(true));
    context.set_abort_flag(abort_flag);

    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert_eq!(result, Err(DecodeError::Aborted));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn set_abort_flag_false_allows_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let abort_flag = Arc::new(AtomicBool::new(false));
    context.set_abort_flag(abort_flag);

    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn clear_abort_callback_allows_decode_with_flag_true(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let abort_flag = Arc::new(AtomicBool::new(true));
    context.set_abort_flag(abort_flag);
    context.clear_abort_callback();

    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn synchronize_completes_without_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.synchronize();

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn detach_threadpool_completes_without_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.detach_threadpool();

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn get_logits_ith_returns_token_not_initialized_for_unknown_index(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let result = context.get_logits_ith(7);

    assert!(matches!(result, Err(LogitsError::TokenNotInitialized(7))));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 64,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn get_logits_ith_returns_token_index_exceeds_context_for_huge_index(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let huge_index = i32::try_from(context.n_ctx())?;
    context.mark_logits_initialized(huge_index);
    let result = context.get_logits_ith(huge_index);

    assert!(matches!(
        result,
        Err(LogitsError::TokenIndexExceedsContext { .. })
    ));

    Ok(())
}

// =========================================================================================
// Group B: Qwen embedding model, embeddings=true. Six embedding-specific tests.
// =========================================================================================

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn decode_with_embeddings_enabled(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_seq_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let embeddings = context.embeddings_seq_ith(0)?;

    assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 4,
    embeddings = true,
)]
fn multi_sequence_embeddings_returns_one_embedding_per_sequence(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let inputs = [
        "alpha is here",
        "beta runs fast",
        "gamma waits",
        "delta jumps",
    ];
    let mut batch = LlamaBatch::new(64, 4)?;

    for (sequence_index, text) in inputs.iter().enumerate() {
        let tokens = fixture.model.str_to_token(text, AddBos::Always)?;
        let sequence_id = i32::try_from(sequence_index)?;

        batch.add_sequence(&tokens, sequence_id, true)?;
    }

    context.decode(&mut batch)?;

    let n_embd = usize::try_from(fixture.model.n_embd())?;
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
/// `agent_embedding_batch_distribution_independent_of_context_size`.
#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 4,
    embeddings = true,
)]
fn embeddings_returns_distinct_values_when_reused_batch_has_extra_capacity(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

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

    let n_embd = usize::try_from(fixture.model.n_embd())?;
    let mut batch = LlamaBatch::new(64, 4)?;
    let mut collected: Vec<Vec<f32>> = Vec::new();

    for iteration_inputs in iterations {
        for (sequence_index, text) in iteration_inputs.iter().enumerate() {
            let tokens = fixture.model.str_to_token(text, AddBos::Always)?;
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

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    let embeddings = context.embeddings_ith(last_index)?;

    assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_ith_returns_null_embedding_error_for_non_embedding_token(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let result = context.embeddings_ith(999);

    assert!(result.is_err());

    Ok(())
}

// =========================================================================================
// Group C: t5-small encoder model, embeddings=true. Single trial.
// =========================================================================================

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn encode_succeeds_with_encoder_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture.model.str_to_token("hello", AddBos::Never)?;
    let mut batch = LlamaBatch::new(512, 1)?;
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&mut batch);

    assert!(result.is_ok());

    Ok(())
}

llama_tests_main!();
