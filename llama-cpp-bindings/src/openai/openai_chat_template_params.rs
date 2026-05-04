/// Parameters for applying OpenAI-compatible chat templates.
#[expect(
    clippy::struct_excessive_bools,
    reason = "this struct mirrors OpenAI API flags which are inherently boolean"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIChatTemplateParams<'params> {
    /// OpenAI-compatible messages JSON array.
    pub messages_json: &'params str,
    /// Optional OpenAI-compatible tools JSON array.
    pub tools_json: Option<&'params str>,
    /// Optional tool choice string.
    pub tool_choice: Option<&'params str>,
    /// Optional JSON schema string for tool grammar generation.
    pub json_schema: Option<&'params str>,
    /// Optional custom grammar string.
    pub grammar: Option<&'params str>,
    /// Optional reasoning format string.
    pub reasoning_format: Option<&'params str>,
    /// Optional chat template kwargs JSON object.
    pub chat_template_kwargs: Option<&'params str>,
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
