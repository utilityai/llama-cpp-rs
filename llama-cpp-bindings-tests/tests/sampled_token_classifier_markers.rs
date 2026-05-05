use anyhow::Result;
use llama_cpp_bindings_tests::TestFixture;

#[test]
fn classifier_resolves_reasoning_markers_for_default_fixture() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let classifier = model.sampled_token_classifier()?;

    assert!(
        classifier.markers().reasoning.is_some(),
        "expected default fixture to expose reasoning markers; got {:?}",
        classifier.markers()
    );

    Ok(())
}

#[test]
fn classifier_resolves_tool_call_diff_runs_without_panic() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let classifier = model.sampled_token_classifier()?;

    let (_no_tools, _with_tools) = model.diagnose_tool_call_synthetic_renders()?;
    let _markers = classifier.markers();

    Ok(())
}
