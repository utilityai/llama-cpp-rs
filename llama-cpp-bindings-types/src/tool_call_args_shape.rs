use crate::bracketed_json_shape::BracketedJsonShape;
use crate::paired_quote_shape::PairedQuoteShape;
use crate::xml_tags_shape::XmlTagsShape;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallArgsShape {
    BracketedJson(BracketedJsonShape),
    PairedQuote(PairedQuoteShape),
    XmlTags(XmlTagsShape),
}
