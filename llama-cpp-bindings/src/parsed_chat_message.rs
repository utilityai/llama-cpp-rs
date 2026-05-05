use crate::parsed_tool_call::ParsedToolCall;

/// Structured view of a parsed assistant turn produced by
/// [`crate::Model::parse_chat_message`]. All fields are owned strings; the
/// raw FFI handle is dropped before this value reaches the caller.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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

    /// True when no content, reasoning, or tool call survived parsing.
    /// Useful for callers that want to short-circuit without inspecting
    /// each field.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
            && self.reasoning_content.is_empty()
            && self.tool_calls.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::ParsedChatMessage;
    use super::ParsedToolCall;

    #[test]
    fn empty_message_reports_empty() {
        let parsed = ParsedChatMessage::default();

        assert!(parsed.is_empty());
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
                "{}".to_owned(),
            )],
        );

        assert!(!parsed.is_empty());
    }

    #[test]
    fn new_preserves_field_order() {
        let parsed = ParsedChatMessage::new(
            "content".to_owned(),
            "thinking".to_owned(),
            vec![ParsedToolCall::new(
                "id".to_owned(),
                "name".to_owned(),
                "{}".to_owned(),
            )],
        );

        assert_eq!(parsed.content, "content");
        assert_eq!(parsed.reasoning_content, "thinking");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "name");
    }
}
