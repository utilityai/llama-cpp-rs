use std::fmt::Debug;
use std::fmt::Display;

pub mod data;
pub mod data_array;
pub mod logit_bias;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LlamaToken(pub llama_cpp_bindings_sys::llama_token);

impl Display for LlamaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LlamaToken {
    #[must_use]
    pub const fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaToken;

    #[test]
    fn display_shows_inner_value() {
        let token = LlamaToken::new(42);
        assert_eq!(format!("{token}"), "42");
    }
}
