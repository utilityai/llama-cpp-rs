use anyhow::Result;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenSection;
use llama_cpp_bindings_tests::TestFixture;

#[test]
fn classifier_starts_in_pending_section_for_default_fixture() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let classifier = model.sampled_token_classifier();

    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
}

#[test]
fn classifier_construction_is_idempotent_across_calls() {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let first = model.sampled_token_classifier();
    let second = model.sampled_token_classifier();

    assert_eq!(first.current_section(), second.current_section());
    assert_eq!(first.usage(), second.usage());
}

#[test]
fn diagnose_tool_call_synthetic_renders_runs_without_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let _ = model.diagnose_tool_call_synthetic_renders()?;

    Ok(())
}
