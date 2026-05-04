use std::collections::HashSet;

use crate::model::{AddBos, ChatTemplateResult, GrammarTriggerType, LlamaModel};
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

use super::grammar_sampler_error::GrammarSamplerError;

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());

    for character in value.chars() {
        match character {
            '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }

    escaped
}

fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }

    let mut anchored = String::new();

    if !pattern.starts_with('^') {
        anchored.push('^');
    }

    anchored.push_str(pattern);

    if !pattern.ends_with('$') {
        anchored.push('$');
    }

    anchored
}

impl ChatTemplateResult {
    /// Builds a grammar sampler from this template result's grammar and trigger configuration.
    ///
    /// Returns `None` if no grammar is present. The returned `HashSet` contains preserved
    /// token IDs that should be decoded with special token handling.
    ///
    /// # Errors
    /// Returns an error if trigger processing or grammar sampler initialization fails.
    pub fn build_grammar_sampler(
        &self,
        model: &LlamaModel,
    ) -> Result<(Option<LlamaSampler>, HashSet<LlamaToken>), GrammarSamplerError> {
        let mut preserved = HashSet::new();

        for token_str in &self.preserved_tokens {
            let tokens = model
                .str_to_token(token_str, AddBos::Never)
                .map_err(|error| GrammarSamplerError::TokenizationFailed(error.to_string()))?;

            if tokens.len() == 1 {
                preserved.insert(tokens[0]);
            }
        }

        let Some(grammar) = self.grammar.as_deref() else {
            return Ok((None, preserved));
        };

        let grammar_sampler = if self.grammar_lazy {
            if self.grammar_triggers.is_empty() {
                return Err(GrammarSamplerError::MissingTriggers);
            }

            let mut trigger_patterns = Vec::new();
            let mut trigger_tokens = Vec::new();

            for trigger in &self.grammar_triggers {
                match trigger.trigger_type {
                    GrammarTriggerType::Token => {
                        if let Some(token) = trigger.token {
                            trigger_tokens.push(token);
                        }
                    }
                    GrammarTriggerType::Word => {
                        let tokens =
                            model
                                .str_to_token(&trigger.value, AddBos::Never)
                                .map_err(|error| {
                                    GrammarSamplerError::TokenizationFailed(error.to_string())
                                })?;

                        if tokens.len() == 1 {
                            if !preserved.contains(&tokens[0]) {
                                return Err(GrammarSamplerError::TriggerWordNotPreserved(
                                    trigger.value.clone(),
                                ));
                            }
                            trigger_tokens.push(tokens[0]);
                        } else {
                            trigger_patterns.push(regex_escape(&trigger.value));
                        }
                    }
                    GrammarTriggerType::Pattern => {
                        trigger_patterns.push(trigger.value.clone());
                    }
                    GrammarTriggerType::PatternFull => {
                        trigger_patterns.push(anchor_pattern(&trigger.value));
                    }
                }
            }

            LlamaSampler::grammar_lazy_patterns(
                model,
                grammar,
                "root",
                &trigger_patterns,
                &trigger_tokens,
            )?
        } else {
            LlamaSampler::grammar(model, grammar, "root")?
        };

        Ok((Some(grammar_sampler), preserved))
    }
}

#[cfg(test)]
mod tests {
    use super::{anchor_pattern, regex_escape};

    #[test]
    fn regex_escape_special_characters() {
        assert_eq!(regex_escape("."), "\\.");
        assert_eq!(regex_escape("^"), "\\^");
        assert_eq!(regex_escape("$"), "\\$");
        assert_eq!(regex_escape("|"), "\\|");
        assert_eq!(regex_escape("("), "\\(");
        assert_eq!(regex_escape(")"), "\\)");
        assert_eq!(regex_escape("*"), "\\*");
        assert_eq!(regex_escape("+"), "\\+");
        assert_eq!(regex_escape("?"), "\\?");
        assert_eq!(regex_escape("["), "\\[");
        assert_eq!(regex_escape("]"), "\\]");
        assert_eq!(regex_escape("{"), "\\{");
        assert_eq!(regex_escape("}"), "\\}");
        assert_eq!(regex_escape("\\"), "\\\\");
    }

    #[test]
    fn regex_escape_normal_text() {
        assert_eq!(regex_escape("hello world"), "hello world");
    }

    #[test]
    fn regex_escape_empty_string() {
        assert_eq!(regex_escape(""), "");
    }

    #[test]
    fn regex_escape_mixed_text() {
        assert_eq!(regex_escape("price: $5.00"), "price: \\$5\\.00");
    }

    #[test]
    fn anchor_pattern_empty_string() {
        assert_eq!(anchor_pattern(""), "^$");
    }

    #[test]
    fn anchor_pattern_already_anchored() {
        assert_eq!(anchor_pattern("^hello$"), "^hello$");
    }

    #[test]
    fn anchor_pattern_needs_start_anchor() {
        assert_eq!(anchor_pattern("hello$"), "^hello$");
    }

    #[test]
    fn anchor_pattern_needs_end_anchor() {
        assert_eq!(anchor_pattern("^hello"), "^hello$");
    }

    #[test]
    fn anchor_pattern_needs_both_anchors() {
        assert_eq!(anchor_pattern("hello"), "^hello$");
    }

    #[cfg(feature = "tests_that_use_llms")]
    mod model_tests {
        use serial_test::serial;

        use crate::model::chat_template_result::ChatTemplateResult;
        use crate::model::grammar_trigger::{GrammarTrigger, GrammarTriggerType};
        use crate::test_model;
        use crate::token::LlamaToken;

        #[test]
        #[serial]
        fn build_grammar_sampler_returns_none_without_grammar() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult::default();
            let (sampler, preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_none());
            assert!(preserved.is_empty());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_returns_sampler_with_non_lazy_grammar() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_without_triggers_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                ..Default::default()
            };
            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_with_word_trigger_multi_token() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Word,
                    value: "function_call".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_with_pattern_trigger() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Pattern,
                    value: "\\{.*".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_with_token_trigger() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Token,
                    value: "tool".to_string(),
                    token: Some(LlamaToken::new(1)),
                }],
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_with_pattern_full_trigger() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::PatternFull,
                    value: "^tool_call$".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_with_preserved_tokens() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                preserved_tokens: vec!["hello".to_string()],
                ..Default::default()
            };
            let (sampler, preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_none());
            assert!(!preserved.is_empty());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_word_trigger_single_token_not_preserved_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Word,
                    value: "\n".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_word_trigger_single_token_preserved() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                preserved_tokens: vec!["\n".to_string()],
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Word,
                    value: "\n".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let (sampler, preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
            assert!(!preserved.is_empty());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_word_trigger_with_null_byte_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Word,
                    value: "null\0byte".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_invalid_grammar_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("this is not a valid grammar at all!!!".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![GrammarTrigger {
                    trigger_type: GrammarTriggerType::Pattern,
                    value: ".*".to_string(),
                    token: None,
                }],
                ..Default::default()
            };
            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_lazy_with_token_trigger_without_token_value() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("root ::= \"hello\"".to_string()),
                grammar_lazy: true,
                grammar_triggers: vec![
                    GrammarTrigger {
                        trigger_type: GrammarTriggerType::Token,
                        value: "tool".to_string(),
                        token: None,
                    },
                    GrammarTrigger {
                        trigger_type: GrammarTriggerType::Pattern,
                        value: ".*".to_string(),
                        token: None,
                    },
                ],
                ..Default::default()
            };
            let (sampler, _preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_some());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_skips_multi_token_strings_leaving_preserved_set_empty() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                preserved_tokens: vec!["hello world this is a long sentence".to_string()],
                ..Default::default()
            };
            let (sampler, preserved) = result.build_grammar_sampler(&model).unwrap();

            assert!(sampler.is_none());
            assert!(preserved.is_empty());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_preserved_token_with_null_byte_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                preserved_tokens: vec!["null\0byte".to_string()],
                grammar: Some("root ::= \"hello\"".to_string()),
                ..ChatTemplateResult::default()
            };

            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }

        #[test]
        #[serial]
        fn build_grammar_sampler_invalid_grammar_returns_error() {
            let (_backend, model) = test_model::load_default_model().unwrap();
            let result = ChatTemplateResult {
                grammar: Some("this is not valid gbnf".to_string()),
                ..ChatTemplateResult::default()
            };

            let build_result = result.build_grammar_sampler(&model);

            assert!(build_result.is_err());
        }
    }
}
