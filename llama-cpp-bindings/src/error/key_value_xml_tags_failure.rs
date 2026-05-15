/// Failures specific to the key-value XML-tags parser (GLM-4.7 `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>`).
#[derive(Debug, thiserror::Error)]
pub enum KeyValueXmlTagsFailure {
    #[error("tool call function tag has empty name")]
    EmptyFunctionName,
    #[error("tool call function block is missing close tag '{expected_close}'")]
    UnclosedFunctionBlock { expected_close: String },
    #[error("tool call function '{function_name}' has key tag with empty content")]
    EmptyKey { function_name: String },
    #[error("tool call function '{function_name}' is missing key close tag '{expected_close}'")]
    UnclosedKeyTag {
        function_name: String,
        expected_close: String,
    },
    #[error(
        "tool call function '{function_name}' key '{key}' is missing value open tag '{expected_open}'"
    )]
    MissingValueTag {
        function_name: String,
        key: String,
        expected_open: String,
    },
    #[error(
        "tool call function '{function_name}' key '{key}' is missing value close tag '{expected_close}'"
    )]
    UnclosedValueTag {
        function_name: String,
        key: String,
        expected_close: String,
    },
}
