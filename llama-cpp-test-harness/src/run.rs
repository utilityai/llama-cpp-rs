use std::process::ExitCode;

use libtest_mimic::Conclusion;

use crate::run_to_conclusions::run_to_conclusions;

fn aggregate_exit_code(conclusions: &[Conclusion]) -> ExitCode {
    if conclusions.iter().any(Conclusion::has_failed) {
        ExitCode::from(101)
    } else {
        ExitCode::SUCCESS
    }
}

#[must_use]
pub fn run() -> ExitCode {
    match run_to_conclusions() {
        Ok(conclusions) => aggregate_exit_code(&conclusions),
        Err(error) => {
            eprintln!("llama-cpp-test-harness: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::ExitCode;

    use libtest_mimic::Conclusion;
    use llama_cpp_bindings::llama_backend::LlamaBackend;

    use crate::harness_run_error::HarnessRunError;
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
    fn run_to_conclusions_errors_when_backend_init_fails() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _hold = LlamaBackend::init().expect("first init must succeed");

        let outcome = run_to_conclusions();

        assert!(matches!(outcome, Err(HarnessRunError::BackendInit(_))));
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
