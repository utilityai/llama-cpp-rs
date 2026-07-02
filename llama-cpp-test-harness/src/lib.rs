#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]

pub mod context_params;
pub mod download_model;
pub mod execution_phase;
pub mod execution_plan;
pub mod harness_arguments_error;
pub mod harness_run_error;
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
