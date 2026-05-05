/// One tool call extracted by [`crate::Model::parse_chat_message`].
///
/// The `arguments_json` field is the raw JSON string emitted by the parser —
/// always a JSON object per OpenAI tool-call conventions, but verifying the
/// shape is the caller's job (typically via a schema validator).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ParsedToolCall {
    #[must_use]
    pub const fn new(id: String, name: String, arguments_json: String) -> Self {
        Self {
            id,
            name,
            arguments_json,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ParsedToolCall;

    #[test]
    fn new_assigns_fields_in_order() {
        let parsed = ParsedToolCall::new(
            "call_1".to_owned(),
            "get_weather".to_owned(),
            "{\"location\":\"Paris\"}".to_owned(),
        );

        assert_eq!(parsed.id, "call_1");
        assert_eq!(parsed.name, "get_weather");
        assert_eq!(parsed.arguments_json, "{\"location\":\"Paris\"}");
    }

    #[test]
    fn default_yields_empty_strings() {
        let parsed = ParsedToolCall::default();

        assert!(parsed.id.is_empty());
        assert!(parsed.name.is_empty());
        assert!(parsed.arguments_json.is_empty());
    }

    #[test]
    fn equal_when_all_fields_match() {
        let left = ParsedToolCall::new("a".to_owned(), "b".to_owned(), "{}".to_owned());
        let right = ParsedToolCall::new("a".to_owned(), "b".to_owned(), "{}".to_owned());

        assert_eq!(left, right);
    }

    #[test]
    fn not_equal_when_arguments_differ() {
        let left =
            ParsedToolCall::new("a".to_owned(), "b".to_owned(), "{\"x\":1}".to_owned());
        let right =
            ParsedToolCall::new("a".to_owned(), "b".to_owned(), "{\"x\":2}".to_owned());

        assert_ne!(left, right);
    }
}
