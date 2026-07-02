use std::process::ExitCode;

use anyhow::Result;
use anyhow::bail;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness::no_op::no_op;
use llama_cpp_test_harness::run_to_conclusions::run_to_conclusions;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    void_logs = true,
)]
fn phase_a_first_passing_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    let formatted = format!("{:?}", fixture.model);
    assert!(formatted.contains("LlamaModel"));
    no_op(fixture)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_a_second_passing_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(fixture.context_params.n_ctx, 512);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_a_intentionally_failing_trial(_fixture: &LlamaFixture<'_>) -> Result<()> {
    bail!("intentional failure to exercise the trial-failure dispatch path");
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_b_first_passing_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(fixture.mtmd_context.is_none());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_b_second_passing_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    let _markers = fixture.model.tool_call_markers();
    Ok(())
}

//

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
#[llama_test(
    model_source = HuggingFace("intentee-test-harness/does-not-exist", "no-such-file.gguf"),
    n_gpu_layers = 0,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 1,
    n_batch = 1,
    n_ubatch = 1
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("intentee-test-harness/does-not-exist", "no-such-mmproj.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = LocalPath("/nonexistent/llama-cpp-test-harness/no-such-mmproj.gguf"),
)]
fn shared_setup_failure_and_phase_a_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(fixture.model_path.exists());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf")
)]
fn phase_d_mmproj_trial(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert!(
        fixture.mtmd_context.is_some(),
        "mmproj_file declared, but fixture.mtmd_context is None",
    );
    Ok(())
}

const EXPECTED_PHASES: usize = 6;
const EXPECTED_PASSED: u64 = 6;
const EXPECTED_FAILED: u64 = 4;

fn main() -> ExitCode {
    let conclusions = match run_to_conclusions() {
        Ok(conclusions) => conclusions,
        Err(error) => {
            eprintln!("harness_self_test: unexpected harness setup failure: {error}");

            return ExitCode::FAILURE;
        }
    };
    let phases = conclusions.len();
    let total_passed: u64 = conclusions
        .iter()
        .map(|conclusion| conclusion.num_passed)
        .sum();
    let total_failed: u64 = conclusions
        .iter()
        .map(|conclusion| conclusion.num_failed)
        .sum();

    if phases == EXPECTED_PHASES
        && total_passed == EXPECTED_PASSED
        && total_failed == EXPECTED_FAILED
    {
        eprintln!(
            "harness_self_test: as expected — phases={phases}, passed={total_passed}, failed={total_failed}"
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "harness_self_test: UNEXPECTED — phases={phases} (want {EXPECTED_PHASES}), \
             passed={total_passed} (want {EXPECTED_PASSED}), \
             failed={total_failed} (want {EXPECTED_FAILED})"
        );
        ExitCode::FAILURE
    }
}
