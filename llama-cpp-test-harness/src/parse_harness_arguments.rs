use libtest_mimic::Arguments;

use crate::harness_arguments_error::HarnessArgumentsError;

fn validate(mut arguments: Arguments) -> Result<Arguments, HarnessArgumentsError> {
    match arguments.test_threads {
        None | Some(1) => {
            arguments.test_threads = Some(1);
            Ok(arguments)
        }
        Some(requested) => Err(HarnessArgumentsError::ConflictingTestThreads { requested }),
    }
}

/// # Errors
///
/// Returns [`HarnessArgumentsError::ConflictingTestThreads`] when `--test-threads` is set to
/// any value other than `1`. The harness orchestrates phase batching itself and cannot share
/// that responsibility with `libtest_mimic`'s thread pool.
pub fn parse_harness_arguments() -> Result<Arguments, HarnessArgumentsError> {
    validate(Arguments::from_args())
}

#[cfg(test)]
mod tests {
    use libtest_mimic::Arguments;

    use crate::harness_arguments_error::HarnessArgumentsError;

    use super::validate;

    #[test]
    fn validate_accepts_unset_test_threads_and_defaults_to_one() {
        let input = Arguments::default();
        let output = validate(input).expect("unset must be accepted");

        assert_eq!(output.test_threads, Some(1));
    }

    #[test]
    fn validate_accepts_explicit_single_thread() {
        let input = Arguments {
            test_threads: Some(1),
            ..Arguments::default()
        };
        let output = validate(input).expect("--test-threads=1 must be accepted");

        assert_eq!(output.test_threads, Some(1));
    }

    #[test]
    fn validate_rejects_non_one_test_threads() {
        let input = Arguments {
            test_threads: Some(8),
            ..Arguments::default()
        };
        let error = validate(input).expect_err("--test-threads=8 must be rejected");

        assert_eq!(
            error,
            HarnessArgumentsError::ConflictingTestThreads { requested: 8 }
        );
    }

    #[test]
    fn validate_preserves_other_settings() {
        let input = Arguments {
            list: true,
            filter: Some("foo".to_owned()),
            ..Arguments::default()
        };
        let output = validate(input).expect("default test_threads must pass");

        assert!(output.list);
        assert_eq!(output.filter.as_deref(), Some("foo"));
    }
}
