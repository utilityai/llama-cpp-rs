use crate::bracketed_json_shape::BracketedJsonShape;
use crate::key_value_xml_tags_shape::KeyValueXmlTagsShape;
use crate::paired_quote_shape::PairedQuoteShape;
use crate::xml_tags_shape::XmlTagsShape;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallArgsShape {
    BracketedJson(BracketedJsonShape),
    KeyValueXmlTags(KeyValueXmlTagsShape),
    PairedQuote(PairedQuoteShape),
    XmlTags(XmlTagsShape),
}
