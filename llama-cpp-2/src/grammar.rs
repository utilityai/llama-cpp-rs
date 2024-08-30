//! The grammar module contains the grammar parser and the grammar struct.
//!
//! This allows creating a llama-cpp grammar. This is essentially a translation of the parser in
//! `common` to rust

use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};

use llama_cpp_sys_2::{llama_grammar, llama_grammar_element, llama_gretype};
use std::ptr::NonNull;
use std::str::FromStr;
use tracing::error;

/// Details of extraneous characters after a rule error.
#[derive(thiserror::Error, Debug)]
#[error("Extraneous chars after rule {name:?}: {chars:?}")]
pub struct ExtraneousCharsAfterRule {
    /// The name of the rule being parsed
    pub name: String,
    /// the extraneous characters
    pub chars: String,
    /// the rest of the input, this is still to be parsed.
    pub rest: String,
}

/// There was an error parsing the grammar.
#[derive(thiserror::Error, Debug)]
#[allow(clippy::module_name_repetitions)]
pub enum GrammarParseError {
    /// There was an unexpected end of input.
    #[error("Unexpected end of input")]
    UnexpectedEndOfInput {
        /// the stage of parsing that was being performed when we ran out of input.
        parse_stage: &'static str,
    },
    /// There was unexpected characters after a rule name but before "::=". There can only be whitespace.
    #[error("Unexpected Chars after name {name:?} and before \"::=\": {chars}")]
    UnexpectedCharsAfterName {
        /// the name of the rule being parsed
        name: String,
        /// the unexpected characters
        chars: String,
    },
    /// There was no "::=" after a rule name.
    #[error("Expected ::= after name {name:?}")]
    ExpectedEqualsAfterName {
        /// the name of the rule being parsed
        name: String,
    },
    /// There was no closing bracket in a nested rule.
    #[error("Expected closing bracket in nested rule {name:?}")]
    MissingClosingBracketInNestedRule {
        /// the name of the rule being parsed
        name: String,
    },
    /// There was no rule before a postfix operator.
    #[error("Missing rule before postfix operator in {name:?}")]
    ExpectedRuleBeforePostfixOperator {
        /// the name of the rule being parsed
        name: String,
    },
    /// There was an incorrect hex size.
    #[error("Expected hex number with size {expected_size}, but number was {actual:?}")]
    IncorrectHexSize {
        /// the expected size of the hex number
        expected_size: usize,
        /// the actual hex number
        actual: String,
    },
    /// An unknown escape character was found.
    #[error("Unknown escape {escape:?}")]
    UnknownEscape {
        /// the unknown character
        escape: char,
    },
    /// Failed to parse hex from a string.
    #[error("Failed to parse hex from {string}: {error}")]
    ParseHexError {
        /// the error that occurred when parsing the hex
        #[source]
        error: std::num::ParseIntError,
        /// the string that was being parsed
        string: String,
    },
    /// there was not space after the name
    // todo: is this actually an error?
    #[error("Missing space after name in {rest:?}")]
    MissingSpaceAfterName {
        /// the rest of the input, this is still to be parsed.
        rest: String,
    },
    /// There was unexpected characters after the rule.
    #[error("{0}")]
    ExtraneousCharsAfterRule(ExtraneousCharsAfterRule),
}

/// A grammar for llama-cpp.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaGrammar {
    parse: ParseState,
    pub(crate) grammar: NonNull<llama_grammar>,
}

impl Clone for LlamaGrammar {
    fn clone(&self) -> Self {
        let grammar = unsafe { llama_cpp_sys_2::llama_grammar_copy(self.grammar.as_ptr()) };
        Self {
            parse: self.parse.clone(),
            grammar: NonNull::new(grammar).expect("copied grammar should never be null"),
        }
    }
}

unsafe impl Send for LlamaGrammar {}

unsafe impl Sync for LlamaGrammar {}

#[allow(clippy::module_name_repetitions)]
impl Debug for LlamaGrammar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGrammar")
            .field("grammar", &self.grammar)
            .field("parse", &self.parse)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ParseState {
    symbol_ids: BTreeMap<String, u32>,
    rules: Vec<Vec<llama_grammar_element>>,
}

impl ParseState {
    fn new() -> Self {
        Self {
            symbol_ids: BTreeMap::new(),
            rules: Vec::new(),
        }
    }

    fn get_symbol_id(&mut self, name: &str) -> u32 {
        let next_id =
            u32::try_from(self.symbol_ids.len()).expect("too many rules (must fit into u32)");
        let result = self.symbol_ids.entry(name.to_string()).or_insert(next_id);
        *result
    }

    fn generate_symbol_id(&mut self, name: &str) -> u32 {
        let next_id =
            u32::try_from(self.symbol_ids.len()).expect("too many rules (must fit into u32)");
        let generated_name = format!("{name}_{next_id}");
        let None = self.symbol_ids.insert(generated_name, next_id) else {
            panic!("Failed to create unique name for {name}");
        };
        next_id
    }

    fn parse_rule<'a>(&mut self, rest: &'a str) -> Result<Option<&'a str>, GrammarParseError> {
        let rest = Self::consume_whitespace_and_comments(rest, true);
        if rest.is_empty() {
            return Ok(None);
        }
        let (name, rest) = Self::parse_name(rest)?;
        let rest = rest.trim_start();
        let rule_id = self.get_symbol_id(name);

        let (after_name, rest) =
            rest.split_once("::=")
                .ok_or_else(|| GrammarParseError::ExpectedEqualsAfterName {
                    name: name.to_string(),
                })?;

        if !after_name.is_empty() {
            return Err(GrammarParseError::UnexpectedCharsAfterName {
                name: name.to_string(),
                chars: after_name.to_string(),
            });
        }

        let rest = self.parse_alternatives(name, rule_id, rest, false)?;

        let Some((after_rule, rest)) = rest.split_once('\n') else {
            return Ok(None);
        };

        if !after_rule.chars().all(char::is_whitespace) {
            return Err(GrammarParseError::ExtraneousCharsAfterRule(
                ExtraneousCharsAfterRule {
                    name: name.to_string(),
                    chars: after_rule.to_string(),
                    rest: rest.to_string(),
                },
            ));
        }

        Ok(Some(rest))
    }

    fn consume_whitespace_and_comments(mut rest: &str, allow_newlines: bool) -> &str {
        loop {
            rest = rest.trim_start_matches(
                |c: char| if allow_newlines { true } else { c != '\n' } && c.is_whitespace(),
            );
            if rest.starts_with('#') {
                rest = rest.split_once('\n').map_or("", |(_comment, rest)| rest);
            } else {
                break;
            }
        }
        rest
    }

    fn parse_alternatives<'a>(
        &mut self,
        name: &str,
        id: u32,
        rest: &'a str,
        nested: bool,
    ) -> Result<&'a str, GrammarParseError> {
        let mut rule = Vec::new();
        let rest = self.parse_sequence(rest.trim_start(), name, &mut rule, nested)?;
        let mut rest = Self::consume_whitespace_and_comments(rest, nested);
        while rest.starts_with('|') {
            rule.push(llama_grammar_element {
                type_: llama_cpp_sys_2::LLAMA_GRETYPE_ALT,
                value: 0,
            });
            rest = Self::consume_whitespace_and_comments(&rest[1..], true);
            rest = self.parse_sequence(rest, name, &mut rule, nested)?;
        }
        rule.push(llama_grammar_element {
            type_: llama_cpp_sys_2::LLAMA_GRETYPE_END,
            value: 0,
        });
        self.add_rule(id, rule);
        Ok(rest)
    }

    fn add_rule(&mut self, id: u32, rule: Vec<llama_grammar_element>) {
        let id = id as usize;
        if self.rules.len() <= id {
            self.rules.resize(id + 1, Vec::new());
        }
        self.rules[id] = rule;
    }

    #[allow(clippy::too_many_lines)]
    fn parse_sequence<'a>(
        &mut self,
        mut rest: &'a str,
        name: &str,
        rule: &mut Vec<llama_grammar_element>,
        nested: bool,
    ) -> Result<&'a str, GrammarParseError> {
        let mut last_sym_start = rule.len();
        while !rest.is_empty() {
            let first_char =
                rest.chars()
                    .next()
                    .ok_or(GrammarParseError::UnexpectedEndOfInput {
                        parse_stage: "sequence",
                    })?;
            if first_char == '"' {
                rest = &rest[1..];
                last_sym_start = rule.len();
                while !rest.starts_with('"') {
                    let (c, r) = Self::parse_char(rest)?;
                    rest = r;
                    rule.push(llama_grammar_element {
                        type_: llama_cpp_sys_2::LLAMA_GRETYPE_CHAR,
                        value: c as _,
                    });
                }
                rest = Self::consume_whitespace_and_comments(&rest[1..], nested);
            } else if first_char == '[' {
                rest = &rest[1..];
                let start_type = if rest.starts_with('^') {
                    rest = &rest[1..];
                    llama_cpp_sys_2::LLAMA_GRETYPE_CHAR_NOT
                } else {
                    llama_cpp_sys_2::LLAMA_GRETYPE_CHAR
                };
                last_sym_start = rule.len();
                while !rest.starts_with(']') {
                    let (c, r) = Self::parse_char(rest)?;
                    rest = r;
                    let gre_type = if last_sym_start < rule.len() {
                        llama_cpp_sys_2::LLAMA_GRETYPE_CHAR_ALT
                    } else {
                        start_type
                    };
                    rule.push(llama_grammar_element {
                        type_: gre_type,
                        value: c as _,
                    });
                    if rest.starts_with('-') && rest.get(1..).is_some_and(|r| !r.starts_with(']')) {
                        let (c, r) = Self::parse_char(&rest[1..])?;
                        rest = r;
                        rule.push(llama_grammar_element {
                            type_: llama_cpp_sys_2::LLAMA_GRETYPE_CHAR_RNG_UPPER,
                            value: c as _,
                        });
                    }
                }
                rest = Self::consume_whitespace_and_comments(&rest[1..], nested);
            } else if first_char.is_alphabetic() {
                let (name, r) = Self::parse_name(rest)?;
                rest = Self::consume_whitespace_and_comments(r, nested);
                let ref_rule_id = self.get_symbol_id(name);
                last_sym_start = rule.len();
                rule.push(llama_grammar_element {
                    type_: llama_cpp_sys_2::LLAMA_GRETYPE_RULE_REF,
                    value: ref_rule_id,
                });
            } else if first_char == '(' {
                rest = rest[1..].trim_start();
                let sub_rule_id = self.generate_symbol_id(name);
                rest = self.parse_alternatives(name, sub_rule_id, rest, true)?;
                last_sym_start = rule.len();
                rule.push(llama_grammar_element {
                    type_: llama_cpp_sys_2::LLAMA_GRETYPE_RULE_REF,
                    value: sub_rule_id,
                });
                if !rest.starts_with(')') {
                    return Err(GrammarParseError::MissingClosingBracketInNestedRule {
                        name: name.to_string(),
                    });
                }
                rest = Self::consume_whitespace_and_comments(&rest[1..], nested);
            } else if first_char == '*' || first_char == '+' || first_char == '?' {
                if last_sym_start == rule.len() {
                    return Err(GrammarParseError::ExpectedRuleBeforePostfixOperator {
                        name: name.to_string(),
                    });
                }
                let sub_rule_id = self.generate_symbol_id(name);
                let mut sub_rule: Vec<llama_grammar_element> =
                    rule.iter().skip(last_sym_start).copied().collect();
                if rest.starts_with(['*', '+']) {
                    sub_rule.push(llama_grammar_element {
                        type_: llama_cpp_sys_2::LLAMA_GRETYPE_RULE_REF,
                        value: sub_rule_id,
                    });
                }
                sub_rule.push(llama_grammar_element {
                    type_: llama_cpp_sys_2::LLAMA_GRETYPE_ALT,
                    value: 0,
                });
                if rest.starts_with('+') {
                    sub_rule.extend(rule.iter().skip(last_sym_start).copied());
                }
                sub_rule.push(llama_grammar_element {
                    type_: llama_cpp_sys_2::LLAMA_GRETYPE_END,
                    value: 0,
                });
                self.add_rule(sub_rule_id, sub_rule);

                rule.truncate(last_sym_start);
                rule.push(llama_grammar_element {
                    type_: llama_cpp_sys_2::LLAMA_GRETYPE_RULE_REF,
                    value: sub_rule_id,
                });

                rest = Self::consume_whitespace_and_comments(&rest[1..], nested);
            } else {
                break;
            }
        }

        Ok(rest)
    }

    fn parse_hex(rest: &str, size: usize) -> Result<(llama_gretype, &str), GrammarParseError> {
        if rest.len() < size {
            return Err(GrammarParseError::IncorrectHexSize {
                expected_size: size,
                actual: rest.to_string(),
            });
        }

        let (hex, rest) = rest.split_at(size);
        let value =
            u32::from_str_radix(hex, 16).map_err(|error| GrammarParseError::ParseHexError {
                string: hex.to_string(),
                error,
            })?;

        Ok((value as llama_gretype, rest))
    }

    fn parse_char(rest: &str) -> Result<(llama_gretype, &str), GrammarParseError> {
        if let Some(rest) = rest.strip_prefix('\\') {
            let Some(escaped) = rest.chars().next() else {
                return Err(GrammarParseError::UnexpectedEndOfInput {
                    parse_stage: "escape char",
                });
            };
            let rest = &rest[escaped.len_utf8()..];
            match escaped {
                'x' => Self::parse_hex(rest, 2),
                'u' => Self::parse_hex(rest, 4),
                'U' => Self::parse_hex(rest, 8),
                't' => Ok((u32::from('\t') as llama_gretype, rest)),
                'r' => Ok((u32::from('\r') as llama_gretype, rest)),
                'n' => Ok((u32::from('\n') as llama_gretype, rest)),
                '\\' => Ok((u32::from('\\') as llama_gretype, rest)),
                '"' => Ok((u32::from('"') as llama_gretype, rest)),
                '[' => Ok((u32::from('[') as llama_gretype, rest)),
                ']' => Ok((u32::from(']') as llama_gretype, rest)),
                c => Err(GrammarParseError::UnknownEscape { escape: c }),
            }
        } else if let Some(c) = rest.chars().next() {
            Ok((u32::from(c) as llama_gretype, &rest[c.len_utf8()..]))
        } else {
            Err(GrammarParseError::UnexpectedEndOfInput {
                parse_stage: "char",
            })
        }
    }

    fn parse_name(rest: &str) -> Result<(&str, &str), GrammarParseError> {
        let name_end = rest
            .find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .ok_or(GrammarParseError::MissingSpaceAfterName {
                rest: rest.to_string(),
            })?;
        let name = &rest[..name_end];
        let rest = &rest[name_end..];
        Ok((name, rest))
    }
}

/// An error that can occur creating a grammar from a string.
#[derive(thiserror::Error, Debug)]
pub enum LlamaGrammarFromStrError {
    /// There was an error parsing the grammar.
    #[error("Failed to parse grammar {0}")]
    ParseError(#[from] GrammarParseError),
    /// Llama-cpp returned null - this can occur for many reasons, but should ideally be caught on
    /// the rust side beforehand.
    #[error("llama-cpp returned null")]
    LlamaCppNullError,
}

impl FromStr for ParseState {
    type Err = GrammarParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parse_state = ParseState::new();
        let mut remaining = Some(s);
        while let Some(str) = remaining {
            remaining = parse_state.parse_rule(str)?;
        }
        Ok(parse_state)
    }
}

impl FromStr for LlamaGrammar {
    type Err = LlamaGrammarFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parse_state = ParseState::from_str(s)?;

        let n_rules = parse_state.rules.len();
        let root_id = parse_state.get_symbol_id("root");
        let mut vec = parse_state
            .rules
            .iter_mut()
            .map(|v| v.as_ptr())
            .collect::<Vec<_>>();
        let rules = vec.as_mut_ptr();

        let grammar =
            unsafe { llama_cpp_sys_2::llama_grammar_init(rules, n_rules, root_id as usize) };

        Ok(Self {
            parse: parse_state,
            grammar: NonNull::new(grammar).ok_or(LlamaGrammarFromStrError::LlamaCppNullError)?,
        })
    }
}

impl Drop for LlamaGrammar {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_grammar_free(self.grammar.as_ptr()) }
    }
}

#[cfg(test)]
mod tests;
