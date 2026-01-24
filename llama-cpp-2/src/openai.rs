/// Parameters for applying OpenAI-compatible chat templates.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIChatTemplateParams<'a> {
    /// OpenAI-compatible messages JSON array.
    pub messages_json: &'a str,
    /// Optional OpenAI-compatible tools JSON array.
    pub tools_json: Option<&'a str>,
    /// Optional tool choice string.
    pub tool_choice: Option<&'a str>,
    /// Optional JSON schema string for tool grammar generation.
    pub json_schema: Option<&'a str>,
    /// Optional custom grammar string.
    pub grammar: Option<&'a str>,
    /// Optional reasoning format string.
    pub reasoning_format: Option<&'a str>,
    /// Optional chat template kwargs JSON object.
    pub chat_template_kwargs: Option<&'a str>,
    /// Whether to add the assistant generation prompt.
    pub add_generation_prompt: bool,
    /// Whether to render templates with Jinja.
    pub use_jinja: bool,
    /// Whether to allow parallel tool calls.
    pub parallel_tool_calls: bool,
    /// Whether thinking blocks are enabled.
    pub enable_thinking: bool,
    /// Whether to add BOS.
    pub add_bos: bool,
    /// Whether to add EOS.
    pub add_eos: bool,
    /// Whether to parse tool calls in responses.
    pub parse_tool_calls: bool,
}
