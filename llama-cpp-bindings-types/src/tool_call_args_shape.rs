use crate::bracketed_json_shape::BracketedJsonShape;
use crate::json_object_shape::JsonObjectShape;
use crate::key_value_xml_tags_shape::KeyValueXmlTagsShape;
use crate::paired_quote_shape::PairedQuoteShape;
use crate::xml_tags_shape::XmlTagsShape;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallArgsShape {
    BracketedJson(BracketedJsonShape),
    JsonObject(JsonObjectShape),
    KeyValueXmlTags(KeyValueXmlTagsShape),
    PairedQuote(PairedQuoteShape),
    XmlTags(XmlTagsShape),
}
