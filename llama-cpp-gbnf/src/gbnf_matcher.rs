use core::ffi::c_char;

use llama_cpp_gbnf_sys::llama_grammar;
use llama_cpp_gbnf_sys::llama_rs_gbnf_accept_str;
use llama_cpp_gbnf_sys::llama_rs_gbnf_clone;
use llama_cpp_gbnf_sys::llama_rs_gbnf_free;
use llama_cpp_gbnf_sys::llama_rs_gbnf_is_accepting;
use llama_cpp_gbnf_sys::llama_rs_gbnf_is_rejected;

pub struct GbnfMatcher {
    initial: *mut llama_grammar,
    current: *mut llama_grammar,
}

impl GbnfMatcher {
    /// # Safety
    ///
    /// `initial` must be a non-null grammar handle produced by the grammar
    /// parser; `GbnfMatcher` takes ownership of it and frees it on drop.
    #[must_use]
    pub unsafe fn new(initial: *mut llama_grammar) -> Self {
        let current = unsafe { llama_rs_gbnf_clone(initial) };

        Self { initial, current }
    }

    pub fn feed_str(&mut self, piece: &str) {
        unsafe {
            llama_rs_gbnf_accept_str(self.current, piece.as_ptr().cast::<c_char>(), piece.len());
        }
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        unsafe { llama_rs_gbnf_is_accepting(self.current) }
    }

    #[must_use]
    pub fn is_rejected(&self) -> bool {
        unsafe { llama_rs_gbnf_is_rejected(self.current) }
    }

    pub fn reset(&mut self) {
        unsafe { llama_rs_gbnf_free(self.current) };
        self.current = unsafe { llama_rs_gbnf_clone(self.initial) };
    }
}

impl Drop for GbnfMatcher {
    fn drop(&mut self) {
        unsafe { llama_rs_gbnf_free(self.current) };
        unsafe { llama_rs_gbnf_free(self.initial) };
    }
}

#[cfg(test)]
mod tests {
    use crate::gbnf_grammar::GbnfGrammar;

    #[test]
    fn incremental_feed_reaches_accepting_only_when_complete() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher();

        matcher.feed_str("ye");
        assert!(!matcher.is_accepting());
        assert!(!matcher.is_rejected());

        matcher.feed_str("s");
        assert!(matcher.is_accepting());
    }

    #[test]
    fn feeding_a_violating_piece_rejects() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher();

        matcher.feed_str("no");
        assert!(matcher.is_rejected());
    }

    #[test]
    fn reset_restores_the_initial_state() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher();

        matcher.feed_str("no");
        assert!(matcher.is_rejected());

        matcher.reset();
        assert!(!matcher.is_rejected());

        matcher.feed_str("yes");
        assert!(matcher.is_accepting());
    }
}
