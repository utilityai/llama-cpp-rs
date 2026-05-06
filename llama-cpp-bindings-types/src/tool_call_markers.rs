use crate::tool_call_args_shape::ToolCallArgsShape;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallMarkers {
    pub open: String,
    pub close: String,
    pub args_shape: ToolCallArgsShape,
}
