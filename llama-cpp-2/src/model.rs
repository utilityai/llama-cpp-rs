//! A safe wrapper around `llama_model`.
use std::ffi::{c_char, CStr, CString};
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;
use std::str::Utf8Error;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, MetaValError, NewLlamaChatMessageError, StringToTokenError,
    TokenToStringError,
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
    pub(crate) lora_adapter: NonNull<llama_cpp_sys_2::llama_adapter_lora>,
}

/// A performance-friendly wrapper around [LlamaModel::chat_template] which is then
/// fed into [LlamaModel::apply_chat_template] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a CString to avoid round-trip conversions
/// within the FFI.
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a llama.cpp [chat template](https://github.com/ggerganov/llama.cpp/blob/8a8c4ceb6050bd9392609114ca56ae6d26f5b8f5/src/llama-chat.cpp#L27-L61)
    /// like "chatml" or "llama3" or an actual Jinja template for llama.cpp to interpret.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the CString into a Rust str reference.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// The Rope type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Norm,
    NeoX,
    MRope,
    Vision,
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
    pub(crate) fn vocab_ptr(&self) -> *const llama_cpp_sys_2::llama_vocab {
        unsafe { llama_cpp_sys_2::llama_model_get_vocab(self.model.as_ptr()) }
    }

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
        let token = unsafe { llama_cpp_sys_2::llama_token_bos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_eos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_token_nl(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token represents the end of generation (end of turn, end of sequence, etc.)
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_2::llama_token_is_eog(self.vocab_ptr(), token.0) }
    }

    /// Get the decoder start token.
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
        let bytes = self.token_to_bytes(token, special)?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Convert single token to bytes.
    ///
    /// # Errors
    /// See [`TokenToStringError`] for more information.
    ///
    /// # Panics
    /// If a [`TokenToStringError::InsufficientBufferSpace`] error returned by
    /// [`Self::token_to_bytes_with_size`] contains a positive nonzero value. This should never
    /// happen.
    pub fn token_to_bytes(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<Vec<u8>, TokenToStringError> {
        match self.token_to_bytes_with_size(token, 8, special, None) {
            Err(TokenToStringError::InsufficientBufferSpace(i)) => self.token_to_bytes_with_size(
                token,
                (-i).try_into().expect("Error buffer size is positive"),
                special,
                None,
            ),
            x => x,
        }
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
        let mut builder: Vec<u8> = Vec::with_capacity(tokens.len() * 4);
        for piece in tokens
            .iter()
            .copied()
            .map(|t| self.token_to_bytes(t, special))
        {
            builder.extend_from_slice(&piece?);
        }
        Ok(String::from_utf8(builder)?)
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
        let mut buffer: Vec<LlamaToken> = Vec::with_capacity(tokens_estimation);

        let c_string = CString::new(str)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        let size = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab_ptr(),
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
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
                    self.vocab_ptr(),
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
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
        Ok(buffer)
    }

    /// Get the type of a token.
    ///
    /// # Panics
    ///
    /// If the token type is not known to this library.
    #[must_use]
    pub fn token_attr(&self, LlamaToken(id): LlamaToken) -> LlamaTokenAttrs {
        let token_type = unsafe { llama_cpp_sys_2::llama_token_get_attr(self.vocab_ptr(), id) };
        LlamaTokenAttrs::try_from(token_type).expect("token type is valid")
    }

    /// Convert a token to a string with a specified buffer size.
    ///
    /// Generally you should use [`LlamaModel::token_to_str`] as it is able to decode tokens with
    /// any length.
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
    /// Generally you should use [`LlamaModel::token_to_bytes`] as it is able to handle tokens of
    /// any length.
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
            return Ok(b"\n".to_vec());
        }

        // unsure what to do with this in the face of the 'special' arg + attr changes
        let attrs = self.token_attr(token);
        if attrs.is_empty()
            || attrs
                .intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Byte | LlamaTokenAttr::Unused)
            || attrs.contains(LlamaTokenAttr::Control)
                && (token == self.token_bos() || token == self.token_eos())
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
                self.vocab_ptr(),
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
        unsafe { llama_cpp_sys_2::llama_n_vocab(self.vocab_ptr()) }
    }

    /// The type of vocab the model was trained on.
    ///
    /// # Panics
    ///
    /// If llama-cpp emits a vocab type that is not known to this library.
    #[must_use]
    pub fn vocab_type(&self) -> VocabType {
        // llama_cpp_sys_2::llama_model_get_vocab
        let vocab_type = unsafe { llama_cpp_sys_2::llama_vocab_type(self.vocab_ptr()) };
        VocabType::try_from(vocab_type).expect("invalid vocab type")
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_n_embd(self.model.as_ptr()) }
    }

    /// Returns the total size of all the tensors in the model in bytes.
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_size(self.model.as_ptr()) }
    }

    /// Returns the number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Returns whether the model is a recurrent network (Mamba, RWKV, etc)
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Returns the number of layers within the model.
    pub fn n_layer(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of attention heads within the model.
    pub fn n_head(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of KV attention heads.
    pub fn n_head_kv(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head_kv(self.model.as_ptr()) })
            .unwrap()
    }

    /// Get metadata value as a string by key name
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str(
                    self.model.as_ptr(),
                    key_ptr,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get the number of metadata key/value pairs
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get metadata key name by index
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_key_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get metadata value as a string by index
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Returns the rope type of the model.
    pub fn rope_type(&self) -> Option<RopeType> {
        match unsafe { llama_cpp_sys_2::llama_model_rope_type(self.model.as_ptr()) } {
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NONE => None,
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NORM => Some(RopeType::Norm),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NEOX => Some(RopeType::NeoX),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_MROPE => Some(RopeType::MRope),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_VISION => Some(RopeType::Vision),
            rope_type => {
                tracing::error!(rope_type = rope_type, "Unexpected rope type from llama.cpp");
                None
            }
        }
    }

    /// Get chat template from model by name. If the name parameter is None, the default chat template will be returned.
    ///
    /// You supply this into [Self::apply_chat_template] to get back a string with the appropriate template
    /// substitution applied to convert a list of messages into a prompt the LLM can use to complete
    /// the chat.
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the llama.cpp template engine.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template by that name
    /// * If the chat template is not a valid [`CString`].
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new);
        let name_ptr = match name_cstr {
            Some(Ok(name)) => name.as_ptr(),
            _ => std::ptr::null(),
        };
        let result =
            unsafe { llama_cpp_sys_2::llama_model_chat_template(self.model.as_ptr(), name_ptr) };

        // Convert result to Rust String if not null
        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };
            let chat_template = CString::new(chat_template_cstr.to_bytes())?;
            Ok(LlamaChatTemplate(chat_template))
        }
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

    /// Load a model from multiple split files.
    ///
    /// This function loads a model that has been split across multiple files. This is useful for
    /// very large models that exceed filesystem limitations or need to be distributed across
    /// multiple storage devices.
    ///
    /// # Arguments
    ///
    /// * `paths` - A slice of paths to the split model files
    /// * `params` - The model parameters
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any of the paths cannot be converted to a C string
    /// - The model fails to load from the splits
    /// - Any path doesn't exist or isn't accessible
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_2::model::{LlamaModel, params::LlamaModelParams};
    /// use llama_cpp_2::llama_backend::LlamaBackend;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = LlamaBackend::init()?;
    /// let params = LlamaModelParams::default();
    /// 
    /// let paths = vec![
    ///     Path::new("model-00001-of-00003.gguf"),
    ///     Path::new("model-00002-of-00003.gguf"),
    ///     Path::new("model-00003-of-00003.gguf"),
    /// ];
    /// 
    /// let model = LlamaModel::load_from_splits(&backend, &paths, &params)?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn load_from_splits(
        _: &LlamaBackend,
        paths: &[impl AsRef<Path>],
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        // Convert paths to C strings
        let c_strings: Vec<CString> = paths
            .iter()
            .map(|p| {
                let path = p.as_ref();
                debug_assert!(path.exists(), "{path:?} does not exist");
                let path_str = path
                    .to_str()
                    .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;
                CString::new(path_str).map_err(LlamaModelLoadError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Create array of pointers to C strings
        let c_ptrs: Vec<*const c_char> = c_strings.iter().map(|s| s.as_ptr()).collect();

        // Load the model from splits
        let llama_model = unsafe {
            llama_cpp_sys_2::llama_model_load_from_splits(
                c_ptrs.as_ptr() as *mut *const c_char,
                c_ptrs.len(),
                params.params,
            )
        };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!("Loaded model from {} splits", paths.len());
        Ok(LlamaModel { model })
    }

    /// Build a split GGUF file path for a specific chunk.
    ///
    /// This utility function creates the standardized filename for a split model chunk
    /// following the pattern: `{prefix}-{split_no:05d}-of-{split_count:05d}.gguf`
    ///
    /// # Arguments
    ///
    /// * `path_prefix` - The base path and filename prefix
    /// * `split_no` - The split number (1-indexed)
    /// * `split_count` - The total number of splits
    ///
    /// # Returns
    ///
    /// Returns the formatted split path as a String
    ///
    /// # Example
    ///
    /// ```
    /// use llama_cpp_2::model::LlamaModel;
    ///
    /// let path = LlamaModel::split_path("/models/llama", 2, 4);
    /// assert_eq!(path, "/models/llama-00002-of-00004.gguf");
    /// ```
    pub fn split_path(path_prefix: &str, split_no: i32, split_count: i32) -> String {
        let mut buffer = vec![0u8; 1024];
        let path_prefix_cstr = CString::new(path_prefix).unwrap_or_else(|_| CString::new("").unwrap());
        let len = unsafe {
            llama_cpp_sys_2::llama_split_path(
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len(),
                path_prefix_cstr.as_ptr(),
                split_no,
                split_count,
            )
        };
        
        if len > 0 && len < buffer.len() as i32 {
            buffer.truncate(len as usize);
            String::from_utf8(buffer).unwrap_or_else(|_| String::new())
        } else {
            String::new()
        }
    }

    /// Extract the path prefix from a split filename.
    ///
    /// This function extracts the base path prefix from a split model filename,
    /// but only if the split_no and split_count match the pattern in the filename.
    ///
    /// # Arguments
    ///
    /// * `split_path` - The full path to a split file
    /// * `split_no` - The expected split number
    /// * `split_count` - The expected total number of splits
    ///
    /// # Returns
    ///
    /// Returns `Some(prefix)` if the split pattern matches, `None` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use llama_cpp_2::model::LlamaModel;
    ///
    /// let prefix = LlamaModel::split_prefix("/models/llama-00002-of-00004.gguf", 2, 4);
    /// assert_eq!(prefix, Some("/models/llama".to_string()));
    ///
    /// // Returns None if the pattern doesn't match
    /// let prefix = LlamaModel::split_prefix("/models/llama-00002-of-00004.gguf", 3, 4);
    /// assert_eq!(prefix, None);
    /// ```
    pub fn split_prefix(split_path: &str, split_no: i32, split_count: i32) -> Option<String> {
        let mut buffer = vec![0u8; 1024];
        let split_path_cstr = CString::new(split_path).ok()?;
        let len = unsafe {
            llama_cpp_sys_2::llama_split_prefix(
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len(),
                split_path_cstr.as_ptr(),
                split_no,
                split_count,
            )
        };
        
        if len > 0 && len < buffer.len() as i32 {
            buffer.truncate(len as usize);
            String::from_utf8(buffer).ok()
        } else {
            None
        }
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
            unsafe { llama_cpp_sys_2::llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr()) };

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
    /// Unlike the llama.cpp apply_chat_template which just randomly uses the ChatML template when given
    /// a null pointer for the template, this requires an explicit template to be specified. If you want to
    /// use "chatml", then just do `LlamaChatTemplate::new("chatml")` or any other model name or template
    /// string.
    ///
    /// Use [Self::chat_template] to retrieve the template baked into the model (this is the preferred
    /// mechanism as using the wrong chat template can result in really unexpected responses from the LLM).
    ///
    /// You probably want to set `add_ass` to true so that the generated template string ends with a the
    /// opening tag of the assistant. If you fail to leave a hanging chat tag, the model will likely generate
    /// one into the output and the output may also have unexpected output aside from that.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff: Vec<u8> = vec![0; message_length * 2];

        // Build our llama_cpp_sys_2 chat messages
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        let tmpl_ptr = tmpl.0.as_ptr();

        let res = unsafe {
            llama_cpp_sys_2::llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<c_char>(),
                buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
            )
        };

        if res > buff.len().try_into().expect("Buffer size exceeds i32::MAX") {
            buff.resize(res.try_into().expect("res is negative"), 0);

            let res = unsafe {
                llama_cpp_sys_2::llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_ass,
                    buff.as_mut_ptr().cast::<c_char>(),
                    buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
                )
            };
            assert_eq!(Ok(res), buff.len().try_into());
        }
        buff.truncate(res.try_into().expect("res is negative"));
        Ok(String::from_utf8(buff)?)
    }
}

/// Generic helper function for extracting string values from the C API
/// This are specifically useful for the the metadata functions, where we pass in a buffer
/// to be populated by a string, not yet knowing if the buffer is large enough.
/// If the buffer was not large enough, we get the correct length back, which can be used to
/// construct a buffer of appropriate size.
fn extract_meta_string<F>(c_function: F, capacity: usize) -> Result<String, MetaValError>
where
    F: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];

    // call the foreign function
    let result = c_function(buffer.as_mut_ptr() as *mut c_char, buffer.len());
    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    // check if the response fit in our buffer
    let returned_len = result as usize;
    if returned_len >= capacity {
        // buffer wasn't large enough, try again with the correct capacity.
        return extract_meta_string(c_function, returned_len + 1);
    }

    // verify null termination
    debug_assert_eq!(
        buffer.get(returned_len),
        Some(&0),
        "should end with null byte"
    );

    // resize, convert, and return
    buffer.truncate(returned_len);
    Ok(String::from_utf8(buffer)?)
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
