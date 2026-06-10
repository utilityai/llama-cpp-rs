#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum GrammarRuntimeError {
    #[error("the grammar parser reached an internal error state: {message}")]
    InternalParserError { message: String },
    #[error("the grammar lexer became too complex: {message}")]
    LexerTooComplex { message: String },
    #[error("the grammar parser became too complex: {message}")]
    ParserTooComplex { message: String },
    #[error("the grammar parser exhausted its maximum token budget: {message}")]
    MaxTokensReached { message: String },
    #[error("the grammar parser panicked during {operation}")]
    Panicked { operation: &'static str },
}
