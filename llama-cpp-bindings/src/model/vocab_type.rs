use crate::model::vocab_type_from_int_error::VocabTypeFromIntError;

/// a rusty equivalent of `llama_vocab_type`
#[repr(u32)]
#[derive(Debug, Eq, Copy, Clone, PartialEq)]
pub enum VocabType {
    /// Byte Pair Encoding
    BPE = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE as _,
    /// Sentence Piece Tokenizer
    SPM = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM as _,
}

impl TryFrom<llama_cpp_bindings_sys::llama_vocab_type> for VocabType {
    type Error = VocabTypeFromIntError;

    fn try_from(value: llama_cpp_bindings_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE => Ok(Self::BPE),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM => Ok(Self::SPM),
            unknown => Err(VocabTypeFromIntError::UnknownValue(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VocabType, VocabTypeFromIntError};

    #[test]
    fn try_from_bpe() {
        let result = VocabType::try_from(llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE);

        assert_eq!(result, Ok(VocabType::BPE));
    }

    #[test]
    fn try_from_spm() {
        let result = VocabType::try_from(llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM);

        assert_eq!(result, Ok(VocabType::SPM));
    }

    #[test]
    fn try_from_unknown_value() {
        let result = VocabType::try_from(99999);

        assert_eq!(result, Err(VocabTypeFromIntError::UnknownValue(99999)));
    }
}
