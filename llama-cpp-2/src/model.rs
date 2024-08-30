//! A safe wrapper around `llama_model`.
use std::ffi::CString;
use std::ffi::CStr;
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, NewLlamaChatMessageError, StringToTokenError, TokenToStringError,
};

pub mod params;

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: NonNull<llama_cpp_sys_2::llama_model>,
}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: NonNull<llama_cpp_sys_2::llama_lora_adapter>,
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// How to determine if we should prepend a bos token to tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddBos {
    /// Add the beginning of stream token to the start of the string.
    Always,
    /// Do not add the beginning of stream token to the start of the string.
    Never,
}

/// How to determine if we should tokenize special tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    /// Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.
    Tokenize,
    /// Treat special and/or control tokens as plaintext.
    Plaintext,
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// get the number of tokens the model was trained on
    ///
    /// # Panics
    ///
    /// If the number of tokens the model was trained on does not fit into an `u32`. This should be impossible on most
    /// platforms due to llama.cpp returning a `c_int` (i32 on most platforms) which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_2::llama_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    pub fn tokens(
        &self,
        special: Special,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| (llama_token, self.token_to_str(llama_token, special)))
    }

    /// Get the beginning of stream token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_bos(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_eos(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_nl(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the decoder start token token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_2::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Convert single token to a string.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    pub fn token_to_str(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        self.token_to_str_with_size(token, 32, special)
    }

    /// Convert single token to bytes.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    pub fn token_to_bytes(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<Vec<u8>, TokenToStringError> {
        self.token_to_bytes_with_size(token, 32, special, None)
    }

    /// Convert a vector of tokens to a single string.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    pub fn tokens_to_str(
        &self,
        tokens: &[LlamaToken],
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let mut builder = String::with_capacity(tokens.len() * 4);
        for str in tokens
            .iter()
            .copied()
            .map(|t| self.token_to_str(t, special))
        {
            builder += &str?;
        }
        Ok(builder)
    }

    /// Convert a string to a Vector of tokens.
    ///
    /// # Errors
    ///
    /// - if [`str`] contains a null byte.
    ///
    /// # Panics
    ///
    /// - if there is more than [`usize::MAX`] [`LlamaToken`]s in [`str`].
    ///
    ///
    /// ```no_run
    /// use llama_cpp_2::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::Path;
    /// use llama_cpp_2::model::AddBos;
    /// let backend = llama_cpp_2::llama_backend::LlamaBackend::init()?;
    /// let model = LlamaModel::load_from_file(&backend, Path::new("path/to/model"), &Default::default())?;
    /// let tokens = model.str_to_token("Hello, World!", AddBos::Always)?;
    /// # Ok(())
    /// # }
    pub fn str_to_token(
        &self,
        str: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };

        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + usize::from(add_bos));
        let mut buffer = Vec::with_capacity(tokens_estimation);

        let c_string = CString::new(str)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        let size = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.model.as_ptr(),
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr(),
                buffer_capacity,
                add_bos,
                true,
            )
        };

        // if we fail the first time we can resize the vector to the correct size and try again. This should never fail.
        // as a result - size is guaranteed to be positive here.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                llama_cpp_sys_2::llama_tokenize(
                    self.model.as_ptr(),
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr(),
                    -size,
                    add_bos,
                    true,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }
        Ok(buffer.into_iter().map(LlamaToken).collect())
    }

    /// Get the type of a token.
    ///
    /// # Panics
    ///
    /// If the token type is not known to this library.
    #[must_use]
    pub fn token_attr(&self, LlamaToken(id): LlamaToken) -> LlamaTokenAttrs {
        let token_type = unsafe { llama_cpp_sys_2::llama_token_get_attr(self.model.as_ptr(), id) };
        LlamaTokenAttrs::try_from(token_type).expect("token type is valid")
    }

    /// Convert a token to a string with a specified buffer size.
    ///
    /// Generally you should use [`LlamaModel::token_to_str`] instead as 8 bytes is enough for most words and
    /// the extra bytes do not really matter.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    /// - the string returend by llama-cpp is not valid utf8.
    ///
    /// # Panics
    ///
    /// - if `buffer_size` does not fit into a [`c_int`].
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    pub fn token_to_str_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let bytes = self.token_to_bytes_with_size(token, buffer_size, special, None)?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Convert a token to bytes with a specified buffer size.
    ///
    /// Generally you should use [`LlamaModel::token_to_bytes`] instead as 8 bytes is enough for most words and
    /// the extra bytes do not really matter.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    ///
    /// # Panics
    ///
    /// - if `buffer_size` does not fit into a [`c_int`].
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    pub fn token_to_bytes_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        if token == self.token_nl() {
            return Ok(String::from("\n").into_bytes());
        }

        // unsure what to do with this in the face of the 'special' arg + attr changes
        let attrs = self.token_attr(token);
        if attrs.contains(LlamaTokenAttr::Control)
            && (token == self.token_bos() || token == self.token_eos())
        {
            return Ok(Vec::new());
        } else if attrs.is_empty()
            || attrs
                .intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Byte | LlamaTokenAttr::Unused)
        {
            return Ok(Vec::new());
        }

        let special = match special {
            Special::Tokenize => true,
            Special::Plaintext => false,
        };

        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |it| i32::from(it.get()));
        let size = unsafe {
            llama_cpp_sys_2::llama_token_to_piece(
                self.model.as_ptr(),
                token.0,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }
    /// The number of tokens the model was trained on.
    ///
    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_n_vocab(self.model.as_ptr()) }
    }

    /// The type of vocab the model was trained on.
    ///
    /// # Panics
    ///
    /// If llama-cpp emits a vocab type that is not known to this library.
    #[must_use]
    pub fn vocab_type(&self) -> VocabType {
        let vocab_type = unsafe { llama_cpp_sys_2::llama_vocab_type(self.model.as_ptr()) };
        VocabType::try_from(vocab_type).expect("invalid vocab type")
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_n_embd(self.model.as_ptr()) }
    }

    /// Get chat template from model.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template
    /// * If the chat template is not a valid [`CString`].
    #[allow(clippy::missing_panics_doc)] // we statically know this will not panic as
    pub fn get_chat_template(&self, buf_size: usize) -> Result<String, ChatTemplateError> {
        // longest known template is about 1200 bytes from llama.cpp
        let chat_temp = CString::new(vec![b'*'; buf_size]).expect("no null");
        let chat_ptr = chat_temp.into_raw();
        let chat_name = CString::new("tokenizer.chat_template").expect("no null bytes");

        let ret = unsafe {
            llama_cpp_sys_2::llama_model_meta_val_str(
                self.model.as_ptr(),
                chat_name.as_ptr(),
                chat_ptr,
                buf_size,
            )
        };

        if ret < 0 {
            return Err(ChatTemplateError::MissingTemplate(ret));
        }

        let template_c = unsafe { CString::from_raw(chat_ptr) };
        let template = template_c.to_str()?;

        let ret: usize = ret.try_into().unwrap();
        if template.len() < ret {
            return Err(ChatTemplateError::BuffSizeError(ret + 1));
        }

        Ok(template.to_owned())
    }

    /// Loads a model from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { llama_cpp_sys_2::llama_load_model_from_file(cstr.as_ptr(), params.params) };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Initializes a lora adapter from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter, LlamaLoraAdapterInitError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter =
            unsafe { llama_cpp_sys_2::llama_lora_adapter_init(self.model.as_ptr(), cstr.as_ptr()) };

        let adapter = NonNull::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// # Errors
    ///
    /// There is many ways this can fail. See [`LlamaContextLoadError`] for more information.
    // we intentionally do not derive Copy on `LlamaContextParams` to allow llama.cpp to change the type to be non-trivially copyable.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context(
        &self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext, LlamaContextLoadError> {
        let context_params = params.context_params;
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(self.model.as_ptr(), context_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(self, context, params.embeddings()))
    }

    /// Apply the models chat template to some messages.
    /// See https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    ///
    /// `tmpl` of None means to use the default template provided by llama.cpp for the model
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: Option<String>,
        chat: Vec<LlamaChatMessage>,
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff: Vec<i8> = vec![0_i8; message_length * 4];

        // Build our llama_cpp_sys_2 chat messages
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        // Set the tmpl pointer
        let tmpl = tmpl.map(CString::new);
        let tmpl_ptr = match &tmpl {
            Some(str) => str.as_ref().map_err(Clone::clone)?.as_ptr(),
            None => std::ptr::null(),
        };

        let formatted_chat = unsafe {
            let res = llama_cpp_sys_2::llama_chat_apply_template(
                self.model.as_ptr(),
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<std::os::raw::c_char>(),
                buff.len() as i32,
            );
            // A buffer twice the size should be sufficient for all models, if this is not the case for a new model, we can increase it
            // The error message informs the user to contact a maintainer
            if res > buff.len() as i32 {
                return Err(ApplyChatTemplateError::BuffSizeError);
            }
            Ok::<String, ApplyChatTemplateError>(CStr::from_ptr(buff.as_mut_ptr()).to_string_lossy().to_string())
        }?;
        Ok(formatted_chat)
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free_model(self.model.as_ptr()) }
    }
}

/// a rusty equivalent of `llama_vocab_type`
#[repr(u32)]
#[derive(Debug, Eq, Copy, Clone, PartialEq)]
pub enum VocabType {
    /// Byte Pair Encoding
    BPE = llama_cpp_sys_2::LLAMA_VOCAB_TYPE_BPE as _,
    /// Sentence Piece Tokenizer
    SPM = llama_cpp_sys_2::LLAMA_VOCAB_TYPE_SPM as _,
}

/// There was an error converting a `llama_vocab_type` to a `VocabType`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`. Contains the int value that was invalid.
    #[error("Unknown Value {0}")]
    UnknownValue(llama_cpp_sys_2::llama_vocab_type),
}

impl TryFrom<llama_cpp_sys_2::llama_vocab_type> for VocabType {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_sys_2::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_2::LLAMA_VOCAB_TYPE_BPE => Ok(VocabType::BPE),
            llama_cpp_sys_2::LLAMA_VOCAB_TYPE_SPM => Ok(VocabType::SPM),
            unknown => Err(LlamaTokenTypeFromIntError::UnknownValue(unknown)),
        }
    }
}
