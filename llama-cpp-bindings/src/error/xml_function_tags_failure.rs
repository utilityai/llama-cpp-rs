/// Failures specific to the XML function-tags parser (Qwen 3.5+ `<function=name><parameter=key>val</parameter></function>`).
#[derive(Debug, thiserror::Error)]
pub enum XmlFunctionTagsFailure {
    #[error("tool call function tag has empty name")]
    EmptyFunctionName,
    #[error("tool call function '{function_name}' is missing close tag '{expected_close}'")]
    UnclosedFunctionBlock {
        function_name: String,
        expected_close: String,
    },
    #[error("tool call function '{function_name}' has parameter with empty name")]
    EmptyParameterName { function_name: String },
    #[error(
        "tool call function '{function_name}' parameter '{parameter_name}' is missing close tag '{expected_close}'"
    )]
    UnclosedParameterBlock {
        function_name: String,
        parameter_name: String,
        expected_close: String,
    },
}
