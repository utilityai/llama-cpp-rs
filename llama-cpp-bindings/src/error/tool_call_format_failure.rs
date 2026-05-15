use crate::error::bracketed_args_failure::BracketedArgsFailure;
use crate::error::json_object_failure::JsonObjectFailure;
use crate::error::key_value_xml_tags_failure::KeyValueXmlTagsFailure;
use crate::error::paired_quote_failure::PairedQuoteFailure;
use crate::error::xml_function_tags_failure::XmlFunctionTagsFailure;

/// Top-level failure for the wrapper-side template-override parsers (one variant per supported shape).
#[derive(Debug, thiserror::Error)]
pub enum ToolCallFormatFailure {
    #[error("bracketed-args fallback parser: {0}")]
    BracketedArgs(#[from] BracketedArgsFailure),
    #[error("json-object fallback parser: {0}")]
    JsonObject(#[from] JsonObjectFailure),
    #[error("key-value-xml-tags fallback parser: {0}")]
    KeyValueXmlTags(#[from] KeyValueXmlTagsFailure),
    #[error("paired-quote fallback parser: {0}")]
    PairedQuote(#[from] PairedQuoteFailure),
    #[error("xml-function-tags fallback parser: {0}")]
    XmlFunctionTags(#[from] XmlFunctionTagsFailure),
}
