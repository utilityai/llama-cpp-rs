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
fn classifier_resolves_tool_call_markers_for_default_fixture() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let classifier = model.sampled_token_classifier()?;

    let (no_tools, with_tools) = model.diagnose_tool_call_synthetic_renders()?;

    assert!(
        classifier.markers().tool_call.is_some(),
        "expected default fixture to expose tool-call markers; got markers={:?}\n--- no_tools ---\n{no_tools}\n--- with_tools ---\n{with_tools}",
        classifier.markers()
    );

    Ok(())
}
