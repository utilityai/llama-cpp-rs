#[macro_export]
macro_rules! llama_tests_main {
    () => {
        fn main() -> ::std::process::ExitCode {
            $crate::run()
        }
    };
}
