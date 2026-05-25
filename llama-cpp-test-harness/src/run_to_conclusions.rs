use std::sync::Arc;

use libtest_mimic::Conclusion;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_plan::ExecutionPlan;
use crate::parse_harness_arguments::parse_harness_arguments;

/// Runs every registered test against its declared model and returns one [`Conclusion`] per phase.
///
/// Self-tests use this entry point to inspect pass/fail counts without surrendering the
/// binary's exit code to libtest-mimic. Initializes the backend; panics with a descriptive
/// message if init fails (that's a programming error in test setup).
///
/// # Panics
///
/// Panics if [`LlamaBackend::init`] fails or if the CLI arguments conflict with the harness's
/// single-thread requirement. The harness is meaningless without a backend or with conflicting
/// thread-count flags; a crash is the loudest possible failure signal.
#[must_use]
pub fn run_to_conclusions() -> Vec<Conclusion> {
    let arguments = match parse_harness_arguments() {
        Ok(arguments) => arguments,
        Err(error) => panic!("llama-cpp-test-harness: {error}"),
    };
    let mut backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(error) => panic!("llama-cpp-test-harness: backend init failed: {error}"),
    };
    let plan = ExecutionPlan::from_inventory();
    if plan.requests_void_logs() {
        backend.void_logs();
    }
    let backend = Arc::new(backend);
    plan.run(&backend, &arguments)
}

#[cfg(test)]
mod tests {
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    use super::run_to_conclusions;

    #[test]
    fn empty_inventory_yields_no_conclusions_and_skips_void_logs() {
        // The lib's own inventory has no #[llama_test] registrations, so
        // ExecutionPlan::from_inventory() returns an empty plan. requests_void_logs() returns
        // false → the `backend.void_logs()` branch is skipped — this test covers that path.
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let conclusions = run_to_conclusions();

        assert!(
            conclusions.is_empty(),
            "empty inventory must yield zero phase conclusions"
        );
    }
}
