use std::ffi::NulError;
use std::string::FromUtf8Error;

#[derive(Debug, thiserror::Error)]
pub enum JsonSchemaToGrammarError {
    #[error("schema string contains an interior NUL byte: {0}")]
    SchemaContainsNulByte(#[from] NulError),
    #[error("llama_rs_json_schema_to_grammar called with null schema_json")]
    NullSchemaJsonArg,
    #[error("llama_rs_json_schema_to_grammar called with null out_grammar")]
    NullOutGrammarArg,
    #[error("llama_rs_json_schema_to_grammar called with null out_error")]
    NullOutErrorArg,
    #[error("wrapper failed to duplicate the C++ exception message into a Rust-owned string")]
    ErrorStringAllocationFailed,
    #[error("llama_rs_json_schema_to_grammar threw a C++ exception: {message}")]
    VendoredThrewCxxException { message: String },
    #[error("grammar string returned by llama_rs_json_schema_to_grammar is not valid UTF-8")]
    GrammarNotUtf8(#[from] FromUtf8Error),
}
