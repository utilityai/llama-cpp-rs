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
    fn both_variants_destructure_to_their_inner_payloads() {
        let outcomes = [
            ChatMessageParseOutcome::Recognized(ParsedChatMessage::new(
                "content".to_owned(),
                "reasoning".to_owned(),
                Vec::new(),
            )),
            ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                tools_json: "[]".to_owned(),
                text: "raw input".to_owned(),
                is_partial: false,
                ffi_error_message: "parser bailed".to_owned(),
            }),
        ];

        let mut saw_recognized = false;
        let mut saw_unrecognized = false;
        for outcome in outcomes {
            match outcome {
                ChatMessageParseOutcome::Recognized(parsed) => {
                    assert_eq!(parsed.content, "content");
                    assert_eq!(parsed.reasoning_content, "reasoning");
                    assert!(parsed.tool_calls.is_empty());
                    saw_recognized = true;
                }
                ChatMessageParseOutcome::Unrecognized(raw) => {
                    assert_eq!(raw.tools_json, "[]");
                    assert_eq!(raw.text, "raw input");
                    assert!(!raw.is_partial);
                    assert_eq!(raw.ffi_error_message, "parser bailed");
                    saw_unrecognized = true;
                }
            }
        }

        assert!(
            saw_recognized && saw_unrecognized,
            "both variants must dispatch through the match"
        );
    }
}
