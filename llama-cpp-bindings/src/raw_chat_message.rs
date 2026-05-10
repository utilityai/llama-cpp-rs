pub struct RawChatMessage {
    pub tools_json: String,
    pub text: String,
    pub is_partial: bool,
    pub ffi_error_message: String,
}

#[cfg(test)]
mod tests {
    use super::RawChatMessage;

    #[test]
    fn carries_tools_json_text_partial_flag_and_ffi_error_message() {
        let raw = RawChatMessage {
            tools_json: "[]".to_owned(),
            text: "hello".to_owned(),
            is_partial: true,
            ffi_error_message: "parser bailed".to_owned(),
        };

        assert_eq!(raw.tools_json, "[]");
        assert_eq!(raw.text, "hello");
        assert!(raw.is_partial);
        assert_eq!(raw.ffi_error_message, "parser bailed");
    }
}
