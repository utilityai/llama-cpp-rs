#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::fmt::{Debug, Formatter};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Debug for llama_grammar_element {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        fn type_to_str(r#type: llama_gretype) -> &'static str {
            match r#type {
                LLAMA_GRETYPE_END => "END",
                LLAMA_GRETYPE_ALT => "ALT",
                LLAMA_GRETYPE_RULE_REF => "RULE_REF",
                LLAMA_GRETYPE_CHAR => "CHAR",
                LLAMA_GRETYPE_CHAR_NOT => "CHAR_NOT",
                LLAMA_GRETYPE_CHAR_RNG_UPPER => "CHAR_RNG_UPPER",
                LLAMA_GRETYPE_CHAR_ALT => "CHAR_ALT",
                _ => "Unknown",
            }
        }

        f.debug_struct("llama_grammar_element")
            .field("type", &type_to_str(self.type_))
            .field("value", &self.value)
            .finish()
    }
}
