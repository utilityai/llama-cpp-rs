use serde::Deserialize;
use serde::Serialize;

use crate::tool_call_arguments::ToolCallArguments;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: ToolCallArguments,
}

impl ParsedToolCall {
    #[must_use]
    pub const fn new(id: String, name: String, arguments: ToolCallArguments) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ParsedToolCall;
    use crate::tool_call_arguments::ToolCallArguments;

    #[test]
    fn new_assigns_fields_in_order() {
        let parsed = ParsedToolCall::new(
            "id-1".to_owned(),
            "tool".to_owned(),
            ToolCallArguments::ValidJson(json!({})),
        );

        assert_eq!(parsed.id, "id-1");
        assert_eq!(parsed.name, "tool");
        assert_eq!(parsed.arguments, ToolCallArguments::ValidJson(json!({})));
    }

    #[test]
    fn default_is_empty_strings_and_invalid_arguments() {
        let parsed = ParsedToolCall::default();

        assert!(parsed.id.is_empty());
        assert!(parsed.name.is_empty());
        assert_eq!(
            parsed.arguments,
            ToolCallArguments::InvalidJson(String::new())
        );
    }
}
