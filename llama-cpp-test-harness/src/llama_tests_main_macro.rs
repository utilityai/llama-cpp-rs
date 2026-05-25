/// Generates a `fn main() -> ExitCode` that dispatches via the harness.
///
/// Place once at module scope in a test binary that uses `#[llama_test(...)]`.
#[macro_export]
macro_rules! llama_tests_main {
    () => {
        fn main() -> ::std::process::ExitCode {
            $crate::run()
        }
    };
}
