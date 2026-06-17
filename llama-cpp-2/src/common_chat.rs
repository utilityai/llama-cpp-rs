//! llama.cpp common-chat template and parser utilities.

pub use crate::model::{
    ChatTemplateResult as CommonChatTemplateResult, GrammarTrigger, GrammarTriggerType,
};
pub use crate::openai::{
    ChatParseStateOaicompat as CommonChatParseState,
    OpenAIChatTemplateParams as CommonChatTemplateParams,
};
