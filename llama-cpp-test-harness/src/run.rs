use std::process::ExitCode;
use std::sync::Arc;

use libtest_mimic::Conclusion;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_plan::ExecutionPlan;
use crate::parse_harness_arguments::parse_harness_arguments;

fn aggregate_exit_code(conclusions: &[Conclusion]) -> ExitCode {
    if conclusions.iter().any(Conclusion::has_failed) {
        ExitCode::from(101)
    } else {
        ExitCode::SUCCESS
    }
}

#[must_use]
pub fn run() -> ExitCode {
    let arguments = match parse_harness_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("llama-cpp-test-harness: {error}");
            return ExitCode::from(2);
        }
    };
    let mut backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("llama-cpp-test-harness: backend init failed: {error}");
            return ExitCode::from(2);
        }
    };
    let plan = ExecutionPlan::from_inventory();
    if plan.requests_void_logs() {
        backend.void_logs();
    }
    let backend = Arc::new(backend);
    aggregate_exit_code(&plan.run(&backend, &arguments))
}

#[cfg(test)]
mod tests {
    use std::process::ExitCode;

    use libtest_mimic::Conclusion;
    use llama_cpp_bindings::llama_backend::LlamaBackend;

    use crate::run_to_conclusions::run_to_conclusions;
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    use super::aggregate_exit_code;
    use super::run;

    fn passing_conclusion() -> Conclusion {
        Conclusion {
            num_filtered_out: 0,
            num_passed: 1,
            num_failed: 0,
            num_ignored: 0,
            num_measured: 0,
        }
    }

    fn failing_conclusion() -> Conclusion {
        Conclusion {
            num_filtered_out: 0,
            num_passed: 0,
            num_failed: 1,
            num_ignored: 0,
            num_measured: 0,
        }
    }

    fn as_u8(code: ExitCode) -> u8 {
        let formatted = format!("{code:?}");
        formatted
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>()
            .parse::<u8>()
            .unwrap_or(255)
    }

    #[test]
    fn aggregate_exit_code_zero_when_all_pass() {
        let code = aggregate_exit_code(&[passing_conclusion(), passing_conclusion()]);

        assert_eq!(as_u8(code), 0);
    }

    #[test]
    fn aggregate_exit_code_non_zero_when_any_fails() {
        let code = aggregate_exit_code(&[passing_conclusion(), failing_conclusion()]);

        assert_eq!(as_u8(code), 101);
    }

    #[test]
    fn aggregate_exit_code_empty_input_succeeds() {
        let code = aggregate_exit_code(&[]);

        assert_eq!(as_u8(code), 0);
    }

    #[test]
    fn run_to_conclusions_panics_when_backend_init_fails() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hold = LlamaBackend::init().expect("first init must succeed");
        let outcome = std::panic::catch_unwind(run_to_conclusions);

        assert!(
            outcome.is_err(),
            "expected panic from re-initialised backend"
        );
    }

    #[test]
    fn run_returns_exit_code_two_when_backend_init_fails() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hold = LlamaBackend::init().expect("first init must succeed");
        let code = run();

        assert_eq!(as_u8(code), 2);
    }
}
