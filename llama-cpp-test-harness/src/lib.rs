//! Declarative, deterministic, phase-batched integration-test harness for `llama-cpp-bindings`.
//!
//! Tests tag their functions with `#[llama_test(model_source = HuggingFace("…", "…"), …)]`
//! (or `model_source = LocalPath("…")` for a local GGUF). The harness groups tests with
//! identical [`LoadKey`]s into [`ExecutionPhase`]s, loads each phase's model exactly once, and
//! runs every test in the phase sequentially against the shared [`LlamaFixture`].
//!
//! See the workspace README and `tests/` directory for usage examples.

pub mod context_params;
pub mod download_model;
pub mod execution_phase;
pub mod execution_plan;
pub mod harness_arguments_error;
pub mod llama_fixture;
pub mod llama_test_fn;
pub mod llama_test_registration;
pub mod llama_tests_main_macro;
pub mod load_key;
pub mod mmproj_source;
pub mod model_load_params;
pub mod model_source;
pub mod no_op;
pub mod parse_harness_arguments;
pub mod phase_state;
pub mod run;
pub mod run_to_conclusions;
#[cfg(test)]
mod test_backend_gate;

pub use crate::context_params::ContextParams;
pub use crate::execution_phase::ExecutionPhase;
pub use crate::execution_plan::ExecutionPlan;
pub use crate::llama_fixture::LlamaFixture;
pub use crate::llama_test_fn::LlamaTestFn;
pub use crate::llama_test_registration::LlamaTestRegistration;
pub use crate::load_key::LoadKey;
pub use crate::mmproj_source::MmprojSource;
pub use crate::model_load_params::ModelLoadParams;
pub use crate::model_source::ModelSource;
pub use crate::no_op::no_op;
pub use crate::phase_state::PhaseState;
pub use crate::run::run;
pub use crate::run_to_conclusions::run_to_conclusions;
pub use llama_cpp_test_harness_macros::llama_test;

#[doc(hidden)]
pub use inventory;
