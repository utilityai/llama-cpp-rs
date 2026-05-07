use serde::Deserialize;
use serde::Serialize;

use crate::parsed_tool_call::ParsedToolCall;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParsedChatMessage {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

impl ParsedChatMessage {
    #[must_use]
    pub const fn new(
        content: String,
        reasoning_content: String,
        tool_calls: Vec<ParsedToolCall>,
    ) -> Self {
        Self {
            content,
            reasoning_content,
            tool_calls,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.content.is_empty() && self.reasoning_content.is_empty() && self.tool_calls.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::ParsedChatMessage;
    use super::ParsedToolCall;
    use crate::tool_call_arguments::ToolCallArguments;

    #[test]
    fn empty_message_reports_empty() {
        assert!(ParsedChatMessage::default().is_empty());
    }

    #[test]
    fn message_with_content_is_not_empty() {
        let parsed = ParsedChatMessage::new("hello".to_owned(), String::new(), Vec::new());

        assert!(!parsed.is_empty());
    }

    #[test]
    fn message_with_reasoning_is_not_empty() {
        let parsed = ParsedChatMessage::new(String::new(), "thinking".to_owned(), Vec::new());

        assert!(!parsed.is_empty());
    }

    #[test]
    fn message_with_tool_call_is_not_empty() {
        let parsed = ParsedChatMessage::new(
            String::new(),
            String::new(),
            vec![ParsedToolCall::new(
                String::new(),
                "tool".to_owned(),
                ToolCallArguments::default(),
            )],
        );

        assert!(!parsed.is_empty());
    }

    #[test]
    fn message_with_all_three_fields_populated_is_not_empty() {
        let parsed = ParsedChatMessage::new(
            "hello".to_owned(),
            "thinking".to_owned(),
            vec![ParsedToolCall::new(
                "id-1".to_owned(),
                "tool".to_owned(),
                ToolCallArguments::default(),
            )],
        );

        assert!(!parsed.is_empty());
        assert_eq!(parsed.content, "hello");
        assert_eq!(parsed.reasoning_content, "thinking");
        assert_eq!(parsed.tool_calls.len(), 1);
    }
}
