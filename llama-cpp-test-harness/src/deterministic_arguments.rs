use libtest_mimic::Arguments;

const fn build_deterministic_arguments(mut arguments: Arguments) -> Arguments {
    arguments.test_threads = Some(1);
    arguments
}

#[must_use]
pub fn deterministic_arguments_from_cli() -> Arguments {
    build_deterministic_arguments(Arguments::from_args())
}

#[cfg(test)]
mod tests {
    use libtest_mimic::Arguments;

    use super::build_deterministic_arguments;

    #[test]
    fn build_deterministic_arguments_forces_test_threads_to_one() {
        let input = Arguments {
            test_threads: Some(8),
            ..Arguments::default()
        };
        let output = build_deterministic_arguments(input);

        assert_eq!(output.test_threads, Some(1));
    }

    #[test]
    fn build_deterministic_arguments_overrides_unset_test_threads() {
        let input = Arguments::default();
        let output = build_deterministic_arguments(input);

        assert_eq!(output.test_threads, Some(1));
    }

    #[test]
    fn build_deterministic_arguments_preserves_other_settings() {
        let input = Arguments {
            list: true,
            filter: Some("foo".to_owned()),
            ..Arguments::default()
        };
        let output = build_deterministic_arguments(input);

        assert!(output.list);
        assert_eq!(output.filter.as_deref(), Some("foo"));
    }
}
