use thiserror::Error;

#[derive(Debug, Eq, Error, PartialEq)]
pub enum HarnessArgumentsError {
    #[error(
        "the test harness requires --test-threads=1 (or unset); got --test-threads={requested}"
    )]
    ConflictingTestThreads { requested: usize },
}
