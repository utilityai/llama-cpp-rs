use crate::tool_call_value_quote::ToolCallValueQuote;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedQuoteShape {
    pub name_args_separator: String,
    pub value_quote: ToolCallValueQuote,
}
