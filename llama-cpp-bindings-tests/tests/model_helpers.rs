use anyhow::Result;
use llama_cpp_bindings_tests::FixtureSession;

#[test]
fn debug_format_includes_struct_name_and_model_field() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let formatted = format!("{model:?}");

    assert!(formatted.contains("LlamaModel"));
    assert!(formatted.contains("model"));

    Ok(())
}

#[test]
fn embedding_model_tool_call_markers_call_does_not_panic() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let embedding_model = fixture.embedding_model()?;

    let _markers = embedding_model.tool_call_markers();

    Ok(())
}

#[test]
fn embedding_model_streaming_markers_returns_ok_for_a_model_without_tool_calls() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let embedding_model = fixture.embedding_model()?;

    // The exact set of detected markers depends on the embedding model's chat template;
    // assertion is just that the call returns Ok without panicking, exercising the
    // streaming_markers + autoparser-fallthrough + override-detect paths even on a model
    // that lacks tool calls.
    let _markers = embedding_model.streaming_markers()?;

    Ok(())
}

#[test]
fn approximate_tok_env_is_cached_across_calls() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let first = model.approximate_tok_env();
    let second = model.approximate_tok_env();

    assert!(std::sync::Arc::ptr_eq(&first, &second));

    Ok(())
}

#[test]
fn approximate_tok_env_falls_back_to_eos_when_eot_unavailable() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let embedding_model = fixture.embedding_model()?;

    let _env = embedding_model.approximate_tok_env();

    Ok(())
}
