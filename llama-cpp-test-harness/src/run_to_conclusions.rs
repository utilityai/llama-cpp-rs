use std::sync::Arc;

use libtest_mimic::Conclusion;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_plan::ExecutionPlan;
use crate::harness_run_error::HarnessRunError;
use crate::parse_harness_arguments::parse_harness_arguments;

/// # Errors
///
/// Returns [`HarnessRunError`] when the CLI arguments conflict with the harness's single-thread
/// requirement or the llama backend cannot be initialised. Surfacing these as a typed error keeps
/// the failure explicit instead of aborting the process with a panic.
pub fn run_to_conclusions() -> Result<Vec<Conclusion>, HarnessRunError> {
    let arguments = parse_harness_arguments()?;
    let mut backend = LlamaBackend::init()?;
    let plan = ExecutionPlan::from_inventory();
    if plan.requests_void_logs() {
        backend.void_logs();
    }
    let backend = Arc::new(backend);

    Ok(plan.run(&backend, &arguments))
}

#[cfg(test)]
mod tests {
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    use super::run_to_conclusions;

    #[test]
    fn empty_inventory_yields_no_conclusions_and_skips_void_logs() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let conclusions =
            run_to_conclusions().expect("empty inventory must run without a setup failure");

        assert!(
            conclusions.is_empty(),
            "empty inventory must yield zero phase conclusions"
        );
    }
}
