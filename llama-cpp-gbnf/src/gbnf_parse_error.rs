use std::ffi::NulError;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfParseError {
    #[error("grammar string contains an interior NUL byte")]
    GrammarStringNulByte(#[source] NulError),
    #[error("grammar root name contains an interior NUL byte")]
    RootNameNulByte(#[source] NulError),
    #[error("grammar has a syntax error and could not be parsed")]
    SyntaxError,
    #[error("grammar defines no rules")]
    EmptyRuleSet,
    #[error("grammar does not define the root symbol {root:?}")]
    RootSymbolMissing { root: String },
    #[error("grammar is left-recursive and cannot be compiled into a matcher")]
    LeftRecursion,
}
