#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyValueXmlTagsShape {
    pub key_open: String,
    pub key_close: String,
    pub value_open: String,
    pub value_close: String,
}
