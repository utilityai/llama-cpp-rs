use anyhow::Result;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
use llama_cpp_bindings::streaming_markers::StreamingMarkers;
use llama_cpp_bindings_tests::FixtureSession;

#[test]
fn classifier_starts_in_pending_section_for_default_fixture() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let classifier = model.sampled_token_classifier();

    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
}

#[test]
fn classifier_construction_is_idempotent_across_calls() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let first = model.sampled_token_classifier();
    let second = model.sampled_token_classifier();

    assert_eq!(first.current_section(), second.current_section());
    assert_eq!(first.usage(), second.usage());
}

#[test]
fn diagnose_tool_call_synthetic_renders_runs_without_panic() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let _ = model.diagnose_tool_call_synthetic_renders()?;

    Ok(())
}

#[test]
fn ingest_with_no_markers_emits_undeterminable_with_visible_and_raw_piece() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());

    let outcomes = classifier.ingest(model.token_bos());

    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert!(matches!(
        outcome.sampled_token,
        SampledToken::Undeterminable(_)
    ));
    assert_eq!(outcome.visible_piece, outcome.raw_piece);
    assert_eq!(classifier.usage().undeterminable_tokens, 1);
}

#[test]
fn ingest_with_no_markers_decodes_each_token_independently() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());

    let _ = classifier.ingest(model.token_bos());
    let _ = classifier.ingest(model.token_eos());

    assert_eq!(classifier.usage().undeterminable_tokens, 2);
}

#[test]
fn ingest_prompt_token_with_no_markers_is_a_noop() {
    let fixture = FixtureSession::open().expect("open fixture");
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let usage_before = *classifier.usage();

    classifier.ingest_prompt_token(model.token_bos());
    classifier.ingest_prompt_tokens(&[model.token_eos(), model.token_nl()]);

    assert_eq!(*classifier.usage(), usage_before);
    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
}

#[test]
fn feed_prompt_to_batch_increments_pending_prompt_tokens() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;
    classifier.feed_prompt_to_batch(&mut batch, model.token_eos(), 1, &[0], false)?;

    assert_eq!(classifier.pending_prompt_tokens(), 2);
    assert_eq!(batch.n_tokens(), 2);

    Ok(())
}

#[test]
fn feed_prompt_sequence_to_batch_stages_all_tokens() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    let tokens = vec![model.token_bos(), model.token_eos(), model.token_nl()];
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), 3);
    assert_eq!(batch.n_tokens(), 3);

    Ok(())
}

#[test]
fn commit_prompt_tokens_promotes_pending_count_to_usage_and_clears() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;
    classifier.feed_prompt_to_batch(&mut batch, model.token_eos(), 1, &[0], false)?;

    let promoted = classifier.commit_prompt_tokens();

    assert_eq!(promoted, 2);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 2);

    Ok(())
}

#[test]
fn discard_pending_prompt_tokens_clears_count_without_recording_usage() -> Result<()> {
    let fixture = FixtureSession::open()?;
    let model = fixture.default_model();

    let mut classifier = SampledTokenClassifier::new(model, StreamingMarkers::default());
    let mut batch = LlamaBatch::new(8, 1)?;

    classifier.feed_prompt_to_batch(&mut batch, model.token_bos(), 0, &[0], false)?;

    let discarded = classifier.discard_pending_prompt_tokens();

    assert_eq!(discarded, 1);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    Ok(())
}
