use std::ffi::CString;
use std::ptr;

use llama_cpp_gbnf_sys::LlamaRsGbnfParseStatus;
use llama_cpp_gbnf_sys::llama_grammar;
use llama_cpp_gbnf_sys::llama_rs_gbnf_clone;
use llama_cpp_gbnf_sys::llama_rs_gbnf_free;
use llama_cpp_gbnf_sys::llama_rs_gbnf_parse;

use crate::gbnf_matcher::GbnfMatcher;
use crate::gbnf_parse_error::GbnfParseError;
use crate::validation::Validation;

fn parse_status_to_result(
    status: LlamaRsGbnfParseStatus,
    grammar: *mut llama_grammar,
    root: &str,
) -> Result<*mut llama_grammar, GbnfParseError> {
    match status {
        LlamaRsGbnfParseStatus::Ok => Ok(grammar),
        LlamaRsGbnfParseStatus::SyntaxError => Err(GbnfParseError::SyntaxError),
        LlamaRsGbnfParseStatus::EmptyRuleSet => Err(GbnfParseError::EmptyRuleSet),
        LlamaRsGbnfParseStatus::RootSymbolMissing => Err(GbnfParseError::RootSymbolMissing {
            root: root.to_owned(),
        }),
        LlamaRsGbnfParseStatus::LeftRecursion => Err(GbnfParseError::LeftRecursion),
    }
}

#[derive(Debug)]
pub struct GbnfGrammar {
    handle: *mut llama_grammar,
}

impl GbnfGrammar {
    /// # Errors
    ///
    /// Returns [`GbnfParseError`] when the grammar or root contains a NUL byte,
    /// or the grammar parser rejects the grammar.
    pub fn parse(grammar: &str, root: &str) -> Result<Self, GbnfParseError> {
        let grammar_cstring =
            CString::new(grammar).map_err(GbnfParseError::GrammarStringNulByte)?;
        let root_cstring = CString::new(root).map_err(GbnfParseError::RootNameNulByte)?;

        let mut handle: *mut llama_grammar = ptr::null_mut();

        let status = unsafe {
            llama_rs_gbnf_parse(
                grammar_cstring.as_ptr(),
                root_cstring.as_ptr(),
                &raw mut handle,
            )
        };

        Ok(Self {
            handle: parse_status_to_result(status, handle, root)?,
        })
    }

    #[must_use]
    pub fn matcher(&self) -> GbnfMatcher {
        let initial = unsafe { llama_rs_gbnf_clone(self.handle) };

        unsafe { GbnfMatcher::new(initial) }
    }

    #[must_use]
    pub fn matches(&self, text: &str) -> Validation {
        let mut matcher = self.matcher();

        matcher.feed_str(text);

        if !matcher.is_rejected() && matcher.is_accepting() {
            Validation::Accepted
        } else {
            Validation::Rejected
        }
    }
}

impl Drop for GbnfGrammar {
    fn drop(&mut self) {
        unsafe { llama_rs_gbnf_free(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::mem::discriminant;

    use super::GbnfGrammar;
    use crate::gbnf_parse_error::GbnfParseError;
    use crate::validation::Validation;

    #[test]
    fn empty_grammar_returns_empty_rule_set() {
        let error = GbnfGrammar::parse("", "root").expect_err("empty grammar is rejected");

        assert_eq!(error, GbnfParseError::EmptyRuleSet);
    }

    #[test]
    fn left_recursive_grammar_returns_left_recursion() {
        let error = GbnfGrammar::parse(r#"root ::= root "x""#, "root")
            .expect_err("left recursion is rejected");

        assert_eq!(error, GbnfParseError::LeftRecursion);
    }

    #[test]
    fn valid_grammar_parses() {
        GbnfGrammar::parse(r#"root ::= "yes" | "no""#, "root").expect("valid grammar parses");
    }

    #[test]
    fn malformed_grammar_returns_syntax_error() {
        let error = GbnfGrammar::parse("root ::= (", "root").expect_err("malformed is rejected");

        assert_eq!(error, GbnfParseError::SyntaxError);
    }

    #[test]
    fn missing_root_returns_root_symbol_missing() {
        let error =
            GbnfGrammar::parse(r#"expr ::= "x""#, "root").expect_err("missing root is rejected");

        assert_eq!(
            error,
            GbnfParseError::RootSymbolMissing {
                root: "root".to_owned()
            }
        );
    }

    #[test]
    fn grammar_with_nul_byte_returns_grammar_string_nul_byte() {
        let error =
            GbnfGrammar::parse("root ::= \"a\0b\"", "root").expect_err("nul byte is rejected");
        let representative =
            GbnfParseError::GrammarStringNulByte(CString::new(b"a\0b".to_vec()).unwrap_err());

        assert_eq!(discriminant(&error), discriminant(&representative));
    }

    #[test]
    fn root_with_nul_byte_returns_root_name_nul_byte() {
        let error =
            GbnfGrammar::parse(r#"root ::= "x""#, "ro\0ot").expect_err("nul byte is rejected");
        let representative =
            GbnfParseError::RootNameNulByte(CString::new(b"ro\0ot".to_vec()).unwrap_err());

        assert_eq!(discriminant(&error), discriminant(&representative));
    }

    #[test]
    fn matches_accepts_text_in_the_language() {
        let grammar = GbnfGrammar::parse(r"root ::= [0-9]+", "root").expect("grammar parses");

        assert_eq!(grammar.matches("123"), Validation::Accepted);
    }

    #[test]
    fn matches_rejects_text_outside_the_language() {
        let grammar = GbnfGrammar::parse(r"root ::= [0-9]+", "root").expect("grammar parses");

        assert_eq!(grammar.matches("12a"), Validation::Rejected);
    }

    #[test]
    fn matches_rejects_incomplete_text() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");

        assert_eq!(grammar.matches("ye"), Validation::Rejected);
    }
}
