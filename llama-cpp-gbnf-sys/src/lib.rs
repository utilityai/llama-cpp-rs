use core::ffi::c_char;

#[repr(C)]
pub struct llama_grammar {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRsGbnfParseStatus {
    Ok = 0,
    SyntaxError,
    EmptyRuleSet,
    RootSymbolMissing,
    LeftRecursion,
}

unsafe extern "C" {
    pub fn llama_rs_gbnf_parse(
        grammar_str: *const c_char,
        grammar_root: *const c_char,
        out_grammar: *mut *mut llama_grammar,
    ) -> LlamaRsGbnfParseStatus;

    pub fn llama_rs_gbnf_accept_str(
        grammar: *mut llama_grammar,
        piece: *const c_char,
        piece_len: usize,
    );

    pub fn llama_rs_gbnf_is_accepting(grammar: *mut llama_grammar) -> bool;

    pub fn llama_rs_gbnf_is_rejected(grammar: *mut llama_grammar) -> bool;

    pub fn llama_rs_gbnf_clone(grammar: *const llama_grammar) -> *mut llama_grammar;

    pub fn llama_rs_gbnf_free(grammar: *mut llama_grammar);
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use super::LlamaRsGbnfParseStatus;
    use super::llama_grammar;
    use super::llama_rs_gbnf_free;
    use super::llama_rs_gbnf_parse;

    #[test]
    fn parse_and_free_links_against_grammar_engine() {
        let grammar = CString::new(r#"root ::= "ok""#).expect("grammar has no NUL");
        let root = CString::new("root").expect("root has no NUL");
        let mut out_grammar: *mut llama_grammar = ptr::null_mut();

        let status =
            unsafe { llama_rs_gbnf_parse(grammar.as_ptr(), root.as_ptr(), &raw mut out_grammar) };

        assert_eq!(status, LlamaRsGbnfParseStatus::Ok);
        assert!(!out_grammar.is_null());

        unsafe { llama_rs_gbnf_free(out_grammar) };
    }
}
