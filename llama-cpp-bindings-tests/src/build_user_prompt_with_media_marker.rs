use anyhow::Result;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::mtmd_default_marker;

/// # Errors
///
/// Forwards chat-template lookup, message construction, and template application errors.
pub fn build_user_prompt_with_media_marker(model: &LlamaModel, question: &str) -> Result<String> {
    let marker = mtmd_default_marker()?;
    let user_content = format!("{marker}{question}");
    let chat_template = model.chat_template(None)?;
    let messages = [LlamaChatMessage::new("user".to_string(), user_content)?];

    Ok(model.apply_chat_template(&chat_template, &messages, true)?)
}
