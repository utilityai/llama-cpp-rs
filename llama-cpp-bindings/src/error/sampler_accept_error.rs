#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SamplerAcceptError {
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("grammar state corrupted during accept: {message}")]
    GrammarStateCorrupted { message: String },
    #[error("the grammar sampler callback failed during accept: {message}")]
    GrammarCallbackFailed { message: String },
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}
