use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LlamaTokenData {
    data: llama_cpp_bindings_sys::llama_token_data,
}

impl LlamaTokenData {
    #[must_use]
    pub const fn new(LlamaToken(id): LlamaToken, logit: f32, p: f32) -> Self {
        Self {
            data: llama_cpp_bindings_sys::llama_token_data { id, logit, p },
        }
    }
    #[must_use]
    pub const fn id(&self) -> LlamaToken {
        LlamaToken(self.data.id)
    }

    #[must_use]
    pub const fn logit(&self) -> f32 {
        self.data.logit
    }

    #[must_use]
    pub const fn p(&self) -> f32 {
        self.data.p
    }

    pub const fn set_id(&mut self, id: LlamaToken) {
        self.data.id = id.0;
    }

    pub const fn set_logit(&mut self, logit: f32) {
        self.data.logit = logit;
    }

    pub const fn set_p(&mut self, p: f32) {
        self.data.p = p;
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaTokenData;
    use crate::token::LlamaToken;

    #[test]
    fn new_stores_all_fields() {
        let token = LlamaToken::new(7);
        let data = LlamaTokenData::new(token, 2.5, 0.8);
        assert_eq!(data.id(), token);
        assert!((data.logit() - 2.5_f32).abs() < f32::EPSILON);
        assert!((data.p() - 0.8_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn set_id_updates_token() {
        let mut data = LlamaTokenData::new(LlamaToken::new(1), 0.0, 0.0);
        data.set_id(LlamaToken::new(42));
        assert_eq!(data.id(), LlamaToken::new(42));
    }

    #[test]
    fn set_logit_updates_logit() {
        let mut data = LlamaTokenData::new(LlamaToken::new(1), 0.0, 0.0);
        data.set_logit(-1.5);
        assert!((data.logit() - (-1.5_f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn set_p_updates_probability() {
        let mut data = LlamaTokenData::new(LlamaToken::new(1), 0.0, 0.0);
        data.set_p(0.95);
        assert!((data.p() - 0.95_f32).abs() < f32::EPSILON);
    }
}
