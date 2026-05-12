#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XmlTagsShape {
    pub function_open_prefix: String,
    pub function_close: String,
    pub parameter_open_prefix: String,
    pub parameter_close: String,
}
