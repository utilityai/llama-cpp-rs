#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApplyChatTemplateError {
    #[error("the model has no vocab")]
    NoVocab,
    #[error("the model's chat template rendered an empty prompt or could not be rendered")]
    TemplateApplicationFailed,
    #[error("not enough memory to render the chat template")]
    NotEnoughMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error("the FFI wrapper returned an unrecognized status code {code}")]
    UnrecognizedStatusCode { code: u32 },
}
