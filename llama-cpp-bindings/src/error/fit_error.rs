#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    #[error("no parameter combination fits available memory")]
    NoFittingMemoryLayout,
    #[error("parameter fitting aborted")]
    Aborted,
    #[error("parameter fitting returned an unknown status code: {code}")]
    UnknownStatus { code: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}
