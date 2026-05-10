use llama_cpp_bindings_types::ParsedChatMessage;

use crate::raw_chat_message::RawChatMessage;

pub enum ChatMessageParseOutcome {
    Recognized(ParsedChatMessage),
    Unrecognized(RawChatMessage),
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::ParsedChatMessage;

    use super::ChatMessageParseOutcome;
    use crate::raw_chat_message::RawChatMessage;

    #[test]
    fn recognized_variant_exposes_parsed_chat_message() {
        let parsed =
            ParsedChatMessage::new("content".to_owned(), "reasoning".to_owned(), Vec::new());
        let outcome = ChatMessageParseOutcome::Recognized(parsed);

        match outcome {
            ChatMessageParseOutcome::Recognized(parsed) => {
                assert_eq!(parsed.content, "content");
                assert_eq!(parsed.reasoning_content, "reasoning");
                assert!(parsed.tool_calls.is_empty());
            }
            ChatMessageParseOutcome::Unrecognized(_) => {
                panic!("expected Recognized variant");
            }
        }
    }

    #[test]
    fn unrecognized_variant_exposes_raw_chat_message() {
        let outcome = ChatMessageParseOutcome::Unrecognized(RawChatMessage {
            tools_json: "[]".to_owned(),
            text: "raw input".to_owned(),
            is_partial: false,
            ffi_error_message: "parser bailed".to_owned(),
        });

        match outcome {
            ChatMessageParseOutcome::Unrecognized(raw) => {
                assert_eq!(raw.tools_json, "[]");
                assert_eq!(raw.text, "raw input");
                assert!(!raw.is_partial);
                assert_eq!(raw.ffi_error_message, "parser bailed");
            }
            ChatMessageParseOutcome::Recognized(_) => {
                panic!("expected Unrecognized variant");
            }
        }
    }
}
