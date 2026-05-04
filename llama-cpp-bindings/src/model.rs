//! A safe wrapper around `llama_model`.
use std::ffi::{CStr, CString, c_char};
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;

fn truncated_buffer_to_string(
    mut buffer: Vec<u8>,
    length: usize,
) -> Result<String, ApplyChatTemplateError> {
    buffer.truncate(length);

    Ok(String::from_utf8(buffer)?)
}

fn validate_string_length_for_tokenizer(length: usize) -> Result<c_int, StringToTokenError> {
    Ok(c_int::try_from(length)?)
}

fn cstring_with_validated_len(str: &str) -> Result<(CString, c_int), StringToTokenError> {
    let c_string = CString::new(str)?;
    let len = validate_string_length_for_tokenizer(c_string.as_bytes().len())?;
    Ok((c_string, len))
}
use std::ptr::{self, NonNull};

use crate::context::LlamaContext;
use crate::context::params::LlamaContextParams;
use crate::llama_backend::LlamaBackend;
use crate::openai::OpenAIChatTemplateParams;
use crate::token::LlamaToken;
use crate::token_type::LlamaTokenAttrs;
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, MetaValError, StringToTokenError, TokenToStringError,
};

pub mod add_bos;
pub mod chat_template_result;
pub mod grammar_trigger;
pub mod llama_chat_message;
pub mod llama_chat_template;
pub mod llama_lora_adapter;
pub mod params;
pub mod rope_type;
pub mod split_mode;
pub mod vocab_type;

pub use add_bos::AddBos;
pub use chat_template_result::ChatTemplateResult;
pub use grammar_trigger::{GrammarTrigger, GrammarTriggerType};
pub use llama_chat_message::LlamaChatMessage;
pub use llama_chat_template::LlamaChatTemplate;
pub use llama_lora_adapter::LlamaLoraAdapter;
pub use rope_type::RopeType;
pub use vocab_type::{LlamaTokenTypeFromIntError, VocabType};

use chat_template_result::{new_empty_chat_template_raw_result, parse_chat_template_raw_result};
use params::LlamaModelParams;

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
pub struct LlamaModel {
    /// Raw pointer to the underlying `llama_model`.
    pub model: NonNull<llama_cpp_bindings_sys::llama_model>,
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// Returns a raw pointer to the model's vocabulary.
    #[must_use]
    pub fn vocab_ptr(&self) -> *const llama_cpp_bindings_sys::llama_vocab {
        unsafe { llama_cpp_bindings_sys::llama_model_get_vocab(self.model.as_ptr()) }
    }

    /// Get the number of tokens the model was trained on.
    ///
    /// # Errors
    ///
    /// Returns an error if the value returned by llama.cpp does not fit into a `u32`.
    pub fn n_ctx_train(&self) -> Result<u32, std::num::TryFromIntError> {
        let n_ctx_train = unsafe { llama_cpp_bindings_sys::llama_n_ctx_train(self.model.as_ptr()) };

        u32::try_from(n_ctx_train)
    }

    /// Get all tokens in the model.
    pub fn tokens(
        &self,
        decode_special: bool,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| {
                let mut decoder = encoding_rs::UTF_8.new_decoder();
                (
                    llama_token,
                    self.token_to_piece(llama_token, &mut decoder, decode_special, None),
                )
            })
    }

    /// Get the beginning of stream token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_bos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_eos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_nl(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token represents the end of generation (end of turn, end of sequence, etc.)
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_token_is_eog(self.vocab_ptr(), token.0) }
    }

    /// Get the decoder start token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_bindings_sys::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the separator token (SEP).
    #[must_use]
    pub fn token_sep(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_vocab_sep(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Convert a string to a Vector of tokens.
    ///
    /// # Errors
    ///
    /// - if [`str`] contains a null byte
    /// - if an integer conversion fails during tokenization
    ///
    ///
    /// ```no_run
    /// use llama_cpp_bindings::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::Path;
    /// use llama_cpp_bindings::model::AddBos;
    /// let backend = llama_cpp_bindings::llama_backend::LlamaBackend::init()?;
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

        let (c_string, c_string_len) = cstring_with_validated_len(str)?;
        let buffer_capacity = c_int::try_from(buffer.capacity())?;

        let size = unsafe {
            llama_cpp_bindings_sys::llama_tokenize(
                self.vocab_ptr(),
                c_string.as_ptr(),
                c_string_len,
                buffer
                    .as_mut_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                buffer_capacity,
                add_bos,
                true,
            )
        };

        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size)?);
            unsafe {
                llama_cpp_bindings_sys::llama_tokenize(
                    self.vocab_ptr(),
                    c_string.as_ptr(),
                    c_string_len,
                    buffer
                        .as_mut_ptr()
                        .cast::<llama_cpp_bindings_sys::llama_token>(),
                    -size,
                    add_bos,
                    true,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size)?;

        // SAFETY: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }

        Ok(buffer)
    }

    /// Get the type of a token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token type is not known to this library.
    pub fn token_attr(
        &self,
        LlamaToken(id): LlamaToken,
    ) -> Result<LlamaTokenAttrs, crate::token_type::LlamaTokenTypeFromIntError> {
        let token_type =
            unsafe { llama_cpp_bindings_sys::llama_token_get_attr(self.vocab_ptr(), id) };

        LlamaTokenAttrs::try_from(token_type)
    }

    /// Convert a token to a string using the underlying llama.cpp `llama_token_to_piece` function.
    ///
    /// This is the new default function for token decoding and provides direct access to
    /// the llama.cpp token decoding functionality without any special logic or filtering.
    ///
    /// Decoding raw string requires using an decoder, tokens from language models may not always map
    /// to full characters depending on the encoding so stateful decoding is required, otherwise partial strings may be lost!
    /// Invalid characters are mapped to REPLACEMENT CHARACTER making the method safe to use even if the model inherently produces
    /// garbage.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    ///
    /// - if the returned size from llama.cpp does not fit into a `usize`
    pub fn token_to_piece(
        &self,
        token: LlamaToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError> {
        let bytes = match self.token_to_piece_bytes(token, 8, special, lstrip) {
            Err(TokenToStringError::InsufficientBufferSpace(required_size)) => {
                let buffer_size: usize = (-required_size).try_into()?;

                self.token_to_piece_bytes(token, buffer_size, special, lstrip)
            }
            other => other,
        }?;

        let mut output_piece = String::with_capacity(bytes.len());
        let (_result, _decoded_size, _had_replacements) =
            decoder.decode_to_string(&bytes, &mut output_piece, false);

        Ok(output_piece)
    }

    /// Raw token decoding to bytes, use if you want to handle the decoding model output yourself
    ///
    /// Convert a token to bytes using the underlying llama.cpp `llama_token_to_piece` function. This is mostly
    /// a thin wrapper around `llama_token_to_piece` function, that handles rust <-> c type conversions while
    /// letting the caller handle errors. For a safer interface returning rust strings directly use `token_to_piece` instead!
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    /// - if an integer conversion fails
    #[allow(clippy::missing_panics_doc)]
    pub fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        // SAFETY: `*` (0x2A) is never `\0`, so CString::new cannot fail here
        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len)?;
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |strip_count| i32::from(strip_count.get()));
        let size = unsafe {
            llama_cpp_bindings_sys::llama_token_to_piece(
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
            error_code if error_code.is_negative() => {
                Err(TokenToStringError::InsufficientBufferSpace(error_code))
            }
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size)?;
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
        unsafe { llama_cpp_bindings_sys::llama_n_vocab(self.vocab_ptr()) }
    }

    /// The type of vocab the model was trained on.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp emits a vocab type that is not known to this library.
    pub fn vocab_type(&self) -> Result<VocabType, LlamaTokenTypeFromIntError> {
        let vocab_type = unsafe { llama_cpp_bindings_sys::llama_vocab_type(self.vocab_ptr()) };

        VocabType::try_from(vocab_type)
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_n_embd(self.model.as_ptr()) }
    }

    /// Returns the total size of all the tensors in the model in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_size(self.model.as_ptr()) }
    }

    /// Returns the number of parameters in the model.
    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Returns whether the model is a recurrent network (Mamba, RWKV, etc)
    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Returns the number of layers within the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer count returned by llama.cpp does not fit into a `u32`.
    pub fn n_layer(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_layer(self.model.as_ptr()) })
    }

    /// Returns the number of attention heads within the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head(self.model.as_ptr()) })
    }

    /// Returns the number of KV attention heads.
    ///
    /// # Errors
    ///
    /// Returns an error if the KV head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head_kv(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head_kv(self.model.as_ptr()) })
    }

    /// Returns whether the model is a hybrid network (Jamba, Granite, Qwen3xx, etc.)
    ///
    /// Hybrid models have both attention layers and recurrent/SSM layers.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    /// Get metadata value as a string by key name
    ///
    /// # Errors
    /// Returns an error if the key is not found or the value is not valid UTF-8.
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_val_str(
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
    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get metadata key name by index
    ///
    /// # Errors
    /// Returns an error if the index is out of range or the key is not valid UTF-8.
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_key_by_index(
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
    ///
    /// # Errors
    /// Returns an error if the index is out of range or the value is not valid UTF-8.
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_val_str_by_index(
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
    #[must_use]
    pub fn rope_type(&self) -> Option<RopeType> {
        let raw = unsafe { llama_cpp_bindings_sys::llama_model_rope_type(self.model.as_ptr()) };

        rope_type::rope_type_from_raw(raw)
    }

    /// Get chat template from model by name. If the name parameter is None, the default chat template will be returned.
    ///
    /// You supply this into [`Self::apply_chat_template`] to get back a string with the appropriate template
    /// substitution applied to convert a list of messages into a prompt the LLM can use to complete
    /// the chat.
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the llama.cpp template engine.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template by that name
    ///
    /// # Panics
    ///
    /// Panics if the C-returned chat template string contains interior null bytes
    /// (should never happen with valid model data).
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new);
        let name_ptr = match name_cstr {
            Some(Ok(name)) => name.as_ptr(),
            _ => ptr::null(),
        };
        let result = unsafe {
            llama_cpp_bindings_sys::llama_model_chat_template(self.model.as_ptr(), name_ptr)
        };

        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };
            let chat_template = CString::new(chat_template_cstr.to_bytes())
                .expect("CStr bytes cannot contain interior null bytes");

            Ok(LlamaChatTemplate(chat_template))
        }
    }

    /// Loads a model from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if a valid UTF-8 path somehow contains interior null bytes (should never happen).
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();

        let path_str = path
            .to_str()
            .ok_or_else(|| LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        if !path.exists() {
            return Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()));
        }

        let cstr = CString::new(path_str)?;
        let llama_model = unsafe {
            llama_cpp_bindings_sys::llama_load_model_from_file(cstr.as_ptr(), params.params)
        };

        let model = match NonNull::new(llama_model) {
            Some(ptr) => ptr,
            None if !path.exists() => {
                return Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()));
            }
            None => return Err(LlamaModelLoadError::NullResult),
        };

        Ok(Self { model })
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

        let path_str = path
            .to_str()
            .ok_or_else(|| LlamaLoraAdapterInitError::PathToStrError(path.to_path_buf()))?;

        if !path.exists() {
            return Err(LlamaLoraAdapterInitError::FileNotFound(path.to_path_buf()));
        }

        let cstr = CString::new(path_str)?;
        let raw_adapter = unsafe {
            llama_cpp_bindings_sys::llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr())
        };

        let Some(adapter) = NonNull::new(raw_adapter) else {
            return Err(LlamaLoraAdapterInitError::NullResult);
        };

        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// # Errors
    ///
    /// There is many ways this can fail. See [`LlamaContextLoadError`] for more information.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "LlamaContextParams may become non-trivially copyable upstream"
    )]
    pub fn new_context<'model>(
        &'model self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'model>, LlamaContextLoadError> {
        let context_params = params.context_params;
        let context = unsafe {
            llama_cpp_bindings_sys::llama_new_context_with_model(
                self.model.as_ptr(),
                context_params,
            )
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(self, context, params.embeddings()))
    }

    /// Apply the models chat template to some messages.
    /// See <https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template>
    ///
    /// Unlike the llama.cpp `apply_chat_template` which just randomly uses the `ChatML` template when given
    /// a null pointer for the template, this requires an explicit template to be specified. If you want to
    /// use "chatml", then just do `LlamaChatTemplate::new("chatml")` or any other model name or template
    /// string.
    ///
    /// Use [`Self::chat_template`] to retrieve the template baked into the model (this is the preferred
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
        let message_length = chat.iter().fold(0, |acc, chat_message| {
            acc + chat_message.role.to_bytes().len() + chat_message.content.to_bytes().len()
        });
        let mut buff: Vec<u8> = vec![0; message_length * 2];

        let chat: Vec<llama_cpp_bindings_sys::llama_chat_message> = chat
            .iter()
            .map(|chat_message| llama_cpp_bindings_sys::llama_chat_message {
                role: chat_message.role.as_ptr(),
                content: chat_message.content.as_ptr(),
            })
            .collect();

        let tmpl_ptr = tmpl.0.as_ptr();

        let buff_len: i32 = buff.len().try_into()?;

        let res = unsafe {
            llama_cpp_bindings_sys::llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<c_char>(),
                buff_len,
            )
        };

        if res > buff_len {
            let required_size: usize = res.try_into()?;
            buff.resize(required_size, 0);

            let new_buff_len: i32 = buff.len().try_into()?;

            let res = unsafe {
                llama_cpp_bindings_sys::llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_ass,
                    buff.as_mut_ptr().cast::<c_char>(),
                    new_buff_len,
                )
            };
            let final_size: usize = res.try_into()?;

            return truncated_buffer_to_string(buff, final_size);
        }

        let final_size: usize = res.try_into()?;

        truncated_buffer_to_string(buff, final_size)
    }

    /// Apply the models chat template to some messages and return an optional tool grammar.
    /// `tools_json` should be an OpenAI-compatible tool definition JSON array string.
    /// `json_schema` should be a JSON schema string.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result contains invalid data.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template_with_tools_oaicompat(
        &self,
        tmpl: &LlamaChatTemplate,
        messages: &[LlamaChatMessage],
        tools_json: Option<&str>,
        json_schema: Option<&str>,
        add_generation_prompt: bool,
    ) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
        let chat: Vec<llama_cpp_bindings_sys::llama_chat_message> = messages
            .iter()
            .map(|chat_message| llama_cpp_bindings_sys::llama_chat_message {
                role: chat_message.role.as_ptr(),
                content: chat_message.content.as_ptr(),
            })
            .collect();

        let tools_cstr = tools_json.map(CString::new).transpose()?;
        let json_schema_cstr = json_schema.map(CString::new).transpose()?;

        let mut raw_result = new_empty_chat_template_raw_result();

        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_apply_chat_template_with_tools_oaicompat(
                self.model.as_ptr(),
                tmpl.0.as_ptr(),
                chat.as_ptr(),
                chat.len(),
                tools_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                json_schema_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                add_generation_prompt,
                &raw mut raw_result,
            )
        };

        let parse_tool_calls = tools_json.is_some_and(|tools| !tools.is_empty());

        unsafe { parse_chat_template_raw_result(rc, &raw mut raw_result, parse_tool_calls) }
    }

    /// Apply the model chat template using OpenAI-compatible JSON messages.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result contains invalid data.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template_oaicompat(
        &self,
        tmpl: &LlamaChatTemplate,
        params: &OpenAIChatTemplateParams<'_>,
    ) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
        let parse_tool_calls = params.parse_tool_calls;
        let messages_cstr = CString::new(params.messages_json)?;
        let tools_cstr = params.tools_json.map(CString::new).transpose()?;
        let tool_choice_cstr = params.tool_choice.map(CString::new).transpose()?;
        let json_schema_cstr = params.json_schema.map(CString::new).transpose()?;
        let grammar_cstr = params.grammar.map(CString::new).transpose()?;
        let reasoning_cstr = params.reasoning_format.map(CString::new).transpose()?;
        let kwargs_cstr = params.chat_template_kwargs.map(CString::new).transpose()?;

        let mut raw_result = new_empty_chat_template_raw_result();

        let ffi_params = llama_cpp_bindings_sys::llama_rs_chat_template_oaicompat_params {
            messages: messages_cstr.as_ptr(),
            tools: tools_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            tool_choice: tool_choice_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            json_schema: json_schema_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            grammar: grammar_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            reasoning_format: reasoning_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            chat_template_kwargs: kwargs_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            add_generation_prompt: params.add_generation_prompt,
            use_jinja: params.use_jinja,
            parallel_tool_calls: params.parallel_tool_calls,
            enable_thinking: params.enable_thinking,
            add_bos: params.add_bos,
            add_eos: params.add_eos,
        };

        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_apply_chat_template_oaicompat(
                self.model.as_ptr(),
                tmpl.0.as_ptr(),
                &raw const ffi_params,
                &raw mut raw_result,
            )
        };

        unsafe { parse_chat_template_raw_result(rc, &raw mut raw_result, parse_tool_calls) }
    }
}

fn extract_meta_string<TCFunction>(
    c_function: TCFunction,
    capacity: usize,
) -> Result<String, MetaValError>
where
    TCFunction: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());

    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    let returned_len = result.cast_unsigned() as usize;

    if returned_len >= capacity {
        return extract_meta_string(c_function, returned_len + 1);
    }

    if buffer.get(returned_len) != Some(&0) {
        return Err(MetaValError::NegativeReturn(-1));
    }

    buffer.truncate(returned_len);

    Ok(String::from_utf8(buffer)?)
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_free_model(self.model.as_ptr()) }
    }
}

#[cfg(test)]
mod extract_meta_string_tests {
    use super::extract_meta_string;
    use crate::MetaValError;

    #[test]
    fn returns_error_when_null_terminator_missing() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer[0] = b'a';
                buffer[1] = b'b';
                buffer[2] = b'c';
                2
            },
            4,
        );

        assert_eq!(result.unwrap_err(), MetaValError::NegativeReturn(-1));
    }

    #[test]
    fn returns_error_for_negative_return_value() {
        let result = extract_meta_string(|_buf_ptr, _buf_len| -5, 4);

        assert_eq!(result.unwrap_err(), MetaValError::NegativeReturn(-5));
    }

    #[test]
    fn returns_error_for_invalid_utf8_data() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer[0] = 0xFF;
                buffer[1] = 0xFE;
                buffer[2] = 0;
                2
            },
            4,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("FromUtf8Error"));
    }

    #[test]
    fn triggers_buffer_resize_when_returned_len_exceeds_capacity() {
        let initial_capacity: usize = 4;
        let length_exceeding_initial_capacity = 10;
        let written_length = 2;
        let call_count = std::cell::Cell::new(0);
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let count = call_count.get();
                call_count.set(count + 1);
                if count == 0 {
                    length_exceeding_initial_capacity
                } else {
                    let buffer =
                        unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                    buffer[0] = b'h';
                    buffer[1] = b'i';
                    buffer[2] = 0;
                    written_length
                }
            },
            initial_capacity,
        );

        assert_eq!(result.unwrap(), "hi");
    }

    #[test]
    fn cstring_with_validated_len_null_byte_returns_error() {
        let result = super::cstring_with_validated_len("null\0byte");

        assert!(result.is_err());
    }

    #[test]
    fn validate_string_length_overflow_returns_error() {
        let result = super::validate_string_length_for_tokenizer(usize::MAX);

        assert!(result.is_err());
    }

    #[test]
    fn truncated_buffer_to_string_with_invalid_utf8_returns_error() {
        let invalid_utf8 = vec![0xff, 0xfe, 0xfd];
        let result = super::truncated_buffer_to_string(invalid_utf8, 3);

        assert!(result.is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use serial_test::serial;

    use super::LlamaModel;
    use crate::llama_backend::LlamaBackend;
    use crate::model::AddBos;
    use crate::model::params::LlamaModelParams;
    use crate::test_model;

    #[test]
    #[serial]
    fn model_loads_with_valid_metadata() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        assert!(model.n_vocab() > 0);
        assert!(model.n_embd() > 0);
        assert!(model.n_params() > 0);
        assert!(model.n_ctx_train().unwrap() > 0);
    }

    #[test]
    #[serial]
    fn special_tokens_exist() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let bos = model.token_bos();
        let eos = model.token_eos();
        assert_ne!(bos, eos);
        assert!(model.is_eog_token(eos));
        assert!(!model.is_eog_token(bos));
    }

    #[test]
    #[serial]
    fn str_to_token_roundtrip() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let tokens = model.str_to_token("hello world", AddBos::Never).unwrap();
        assert!(!tokens.is_empty());
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let piece = model
            .token_to_piece(tokens[0], &mut decoder, false, None)
            .unwrap();
        assert!(!piece.is_empty());
    }

    #[test]
    #[serial]
    fn chat_template_returns_non_empty() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None);
        assert!(template.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_produces_prompt() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let prompt = model.apply_chat_template(&template, &[message], true);
        assert!(prompt.is_ok());
        assert!(!prompt.unwrap().is_empty());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_produces_result() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: Some("none"),
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);
        assert!(result.is_ok());
        assert!(!result.unwrap().prompt.is_empty());
    }

    #[test]
    #[serial]
    fn meta_count_returns_positive() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        assert!(model.meta_count() > 0);
    }

    #[test]
    #[serial]
    fn tokens_iterator_produces_valid_entries() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let mut count = 0;

        for (token, _piece_result) in model.tokens(false) {
            assert!(token.0 >= 0);
            count += 1;

            if count >= 100 {
                break;
            }
        }

        assert_eq!(count, 100);
    }

    #[test]
    #[serial]
    fn token_to_piece_bytes_returns_bytes_for_known_token() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let tokens = model.str_to_token("hello", AddBos::Never).unwrap();
        let bytes = model
            .token_to_piece_bytes(tokens[0], 32, false, None)
            .unwrap();

        assert!(!bytes.is_empty());
    }

    #[test]
    #[serial]
    fn n_layer_returns_positive() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        assert!(model.n_layer().unwrap() > 0);
    }

    #[test]
    #[serial]
    fn n_head_returns_positive() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        assert!(model.n_head().unwrap() > 0);
    }

    #[test]
    #[serial]
    fn n_head_kv_returns_positive() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        assert!(model.n_head_kv().unwrap() > 0);
    }

    #[test]
    #[serial]
    fn is_hybrid_returns_bool_for_test_model() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        let _ = model.is_hybrid();
    }

    #[test]
    #[serial]
    fn meta_key_by_index_returns_valid_key() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let key = model.meta_key_by_index(0).unwrap();

        assert!(!key.is_empty());
    }

    #[test]
    #[serial]
    fn meta_val_str_by_index_returns_valid_value() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let value = model.meta_val_str_by_index(0).unwrap();

        assert!(!value.is_empty());
    }

    #[test]
    #[serial]
    fn meta_key_by_index_out_of_range_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = model.meta_key_by_index(999_999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn meta_val_str_by_index_out_of_range_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = model.meta_val_str_by_index(999_999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn meta_val_str_returns_value_for_known_key() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let first_key = model.meta_key_by_index(0).unwrap();
        let value = model.meta_val_str(&first_key).unwrap();

        assert!(!value.is_empty());
    }

    #[test]
    #[serial]
    fn model_size_returns_nonzero() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        assert!(model.size() > 0);
    }

    #[test]
    #[serial]
    fn is_recurrent_returns_false_for_transformer() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        assert!(!model.is_recurrent());
    }

    #[test]
    #[serial]
    fn rope_type_does_not_panic() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let _rope_type = model.rope_type();
    }

    #[test]
    #[serial]
    fn load_model_with_invalid_path_returns_error() {
        let backend = LlamaBackend::init().unwrap();
        let model_params = LlamaModelParams::default();
        let result = LlamaModel::load_from_file(&backend, "/nonexistent/model.gguf", &model_params);

        assert_eq!(
            result.unwrap_err(),
            crate::LlamaModelLoadError::FileNotFound(std::path::PathBuf::from(
                "/nonexistent/model.gguf"
            ))
        );
    }

    #[test]
    #[serial]
    fn load_model_with_invalid_file_content_returns_null_result() {
        let backend = LlamaBackend::init().unwrap();
        let model_params = LlamaModelParams::default();
        let dummy_path = std::env::temp_dir().join("llama_test_invalid_model.gguf");
        std::fs::write(&dummy_path, b"not a valid gguf model file").unwrap();

        let result = LlamaModel::load_from_file(&backend, &dummy_path, &model_params);

        assert_eq!(result.unwrap_err(), crate::LlamaModelLoadError::NullResult);
        let _ = std::fs::remove_file(&dummy_path);
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn load_model_with_non_utf8_path_returns_path_to_str_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let backend = LlamaBackend::init().unwrap();
        let model_params = LlamaModelParams::default();
        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

        let result = LlamaModel::load_from_file(&backend, non_utf8_path, &model_params);

        assert_eq!(
            result.unwrap_err(),
            crate::LlamaModelLoadError::PathToStrError(non_utf8_path.to_path_buf())
        );
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn lora_adapter_init_with_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (_backend, model) = test_model::load_default_model().unwrap();
        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));

        let result = model.lora_adapter_init(non_utf8_path);

        assert_eq!(
            result.unwrap_err(),
            crate::LlamaLoraAdapterInitError::PathToStrError(non_utf8_path.to_path_buf())
        );
    }

    #[test]
    #[serial]
    fn lora_adapter_init_with_invalid_path_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = model.lora_adapter_init("/nonexistent/path/lora.gguf");

        assert_eq!(
            result.unwrap_err(),
            crate::LlamaLoraAdapterInitError::FileNotFound(std::path::PathBuf::from(
                "/nonexistent/path/lora.gguf"
            ))
        );
    }

    #[test]
    #[serial]
    fn new_context_returns_valid_context() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(256));
        let context = model.new_context(&backend, ctx_params).unwrap();

        assert!(context.n_ctx() > 0);
    }

    #[test]
    #[serial]
    fn token_nl_returns_valid_token() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let nl_token = model.token_nl();

        assert!(nl_token.0 >= 0);
    }

    #[test]
    #[serial]
    fn decode_start_token_returns_valid_token() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let _decode_start = model.decode_start_token();
    }

    #[test]
    #[serial]
    fn token_sep_returns_valid_token() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let _sep_token = model.token_sep();
    }

    #[test]
    #[serial]
    fn token_to_piece_handles_large_token_requiring_buffer_resize() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for (token, _) in model.tokens(true).take(200) {
            let result = model.token_to_piece(token, &mut decoder, true, None);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[serial]
    fn token_to_piece_bytes_insufficient_buffer_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let tokens = model.str_to_token("hello", AddBos::Never).unwrap();
        let result = model.token_to_piece_bytes(tokens[0], 1, false, None);

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Insufficient Buffer Space")
        );
    }

    #[test]
    #[serial]
    fn token_to_piece_with_lstrip() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let tokens = model.str_to_token("hello", AddBos::Never).unwrap();
        let result =
            model.token_to_piece(tokens[0], &mut decoder, false, std::num::NonZeroU16::new(1));

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn n_vocab_matches_tokens_iterator_count() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let n_vocab = model.n_vocab();
        let count = model.tokens(false).count();

        assert_eq!(count, n_vocab as usize);
    }

    #[test]
    #[serial]
    fn token_attr_returns_valid_attr() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let bos = model.token_bos();
        let _attr = model.token_attr(bos).unwrap();
    }

    #[test]
    #[serial]
    fn vocab_type_returns_valid_type() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let _vocab_type = model.vocab_type().unwrap();
    }

    #[test]
    #[serial]
    fn apply_chat_template_buffer_resize_with_long_messages() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let long_content = "a".repeat(2000);
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), long_content).unwrap();
        let prompt = model.apply_chat_template(&template, &[message], true);

        assert!(prompt.is_ok());
        assert!(!prompt.unwrap().is_empty());
    }

    #[test]
    #[serial]
    fn meta_val_str_with_long_value_triggers_buffer_resize() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let count = model.meta_count();

        for index in 0..count {
            let key = model.meta_key_by_index(index);
            let value = model.meta_val_str_by_index(index);
            assert!(key.is_ok());
            assert!(value.is_ok());
        }
    }

    #[test]
    #[serial]
    fn str_to_token_with_add_bos_never() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let tokens_with_bos = model.str_to_token("hello", AddBos::Always).unwrap();
        let tokens_without_bos = model.str_to_token("hello", AddBos::Never).unwrap();

        assert!(tokens_with_bos.len() >= tokens_without_bos.len());
    }

    #[test]
    #[serial]
    fn apply_chat_template_with_tools_oaicompat_produces_result() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let result =
            model.apply_chat_template_with_tools_oaicompat(&template, &[message], None, None, true);

        assert!(result.is_ok());
        assert!(!result.unwrap().prompt.is_empty());
    }

    #[test]
    #[serial]
    fn apply_chat_template_with_tools_oaicompat_with_tools_json() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let tools =
            r#"[{"type":"function","function":{"name":"test","parameters":{"type":"object"}}}]"#;
        let result = model.apply_chat_template_with_tools_oaicompat(
            &template,
            &[message],
            Some(tools),
            None,
            true,
        );

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_with_tools_oaicompat_with_json_schema() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
        let result = model.apply_chat_template_with_tools_oaicompat(
            &template,
            &[message],
            None,
            Some(schema),
            true,
        );

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_tools_and_tool_choice() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: Some(
                r#"[{"type":"function","function":{"name":"test","parameters":{"type":"object","properties":{}}}}]"#,
            ),
            tool_choice: Some("auto"),
            json_schema: None,
            grammar: None,
            reasoning_format: Some("none"),
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: true,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_json_schema_field() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: Some(r#"{"type":"object","properties":{"name":{"type":"string"}}}"#),
            grammar: None,
            reasoning_format: Some("none"),
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_grammar_field() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: Some("root ::= \"hello\""),
            reasoning_format: Some("none"),
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_kwargs_field() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: Some("none"),
            chat_template_kwargs: Some(r#"{"bos_token": "<|im_start|>"}"#),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn chat_template_with_nonexistent_name_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        let result = model.chat_template(Some("nonexistent_template_name_xyz"));

        assert_eq!(
            result.unwrap_err(),
            crate::ChatTemplateError::MissingTemplate
        );
    }

    #[test]
    #[serial]
    fn lora_adapter_init_with_invalid_gguf_returns_null_result() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let dummy_path = std::env::temp_dir().join("llama_test_dummy_lora.gguf");
        std::fs::write(&dummy_path, b"not a valid gguf").unwrap();

        let result = model.lora_adapter_init(&dummy_path);

        assert_eq!(
            result.unwrap_err(),
            crate::LlamaLoraAdapterInitError::NullResult
        );
        let _ = std::fs::remove_file(&dummy_path);
    }

    #[test]
    #[serial]
    fn str_to_token_with_many_tokens_triggers_buffer_resize() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let many_numbers: String = (0..2000).map(|number| format!("{number} ")).collect();

        let tokens = model.str_to_token(&many_numbers, AddBos::Always).unwrap();

        assert!(tokens.len() > many_numbers.len() / 2);
    }

    #[test]
    #[serial]
    fn rope_type_returns_valid_result_for_test_model() {
        let (_backend, model) = test_model::load_default_model().unwrap();

        let _rope_type = model.rope_type();
    }

    #[test]
    #[serial]
    fn meta_val_str_with_null_byte_in_key_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = model.meta_val_str("key\0with_null");

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_with_tools_null_byte_in_tools_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let result = model.apply_chat_template_with_tools_oaicompat(
            &template,
            &[message],
            Some("tools\0null"),
            None,
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_with_tools_null_byte_in_json_schema_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let message =
            crate::model::LlamaChatMessage::new("user".to_string(), "hello".to_string()).unwrap();
        let result = model.apply_chat_template_with_tools_oaicompat(
            &template,
            &[message],
            None,
            Some("schema\0null"),
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_messages_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: "messages\0null",
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_tools_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: Some("tools\0null"),
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_tool_choice_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: Some("choice\0null"),
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_json_schema_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: Some("schema\0null"),
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_grammar_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: Some("grammar\0null"),
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_reasoning_format_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: Some("format\0null"),
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn apply_chat_template_oaicompat_with_null_byte_in_kwargs_returns_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let template = model.chat_template(None).unwrap();
        let params = crate::openai::OpenAIChatTemplateParams {
            messages_json: r#"[{"role":"user","content":"hello"}]"#,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: Some("kwargs\0null"),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = model.apply_chat_template_oaicompat(&template, &params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn new_context_with_huge_ctx_returns_null_error() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(u32::MAX));

        let result = model.new_context(&_backend, ctx_params);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn sample_returns_result_and_succeeds_with_valid_index() {
        use crate::sampling::LlamaSampler;
        use crate::token::LlamaToken;

        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(256));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Hello", AddBos::Always).unwrap();
        let mut batch = crate::llama_batch::LlamaBatch::new(512, 1).unwrap();

        batch.add_sequence(&tokens, 0, false).unwrap();

        context.decode(&mut batch).unwrap();

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);

        let result = sampler.sample(&context, batch.n_tokens() - 1);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn grammar_sampler_constrains_output_to_yes_or_no() {
        use crate::sampling::LlamaSampler;
        use std::sync::Arc;

        let (backend, model) = test_model::load_default_model().unwrap();

        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let prompt = "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();
        let mut batch = crate::llama_batch::LlamaBatch::new(512, 1).unwrap();

        batch.add_sequence(&tokens, 0, false).unwrap();

        context.decode(&mut batch).unwrap();

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::grammar(&model, r#"root ::= [Yy] [Ee] [Ss] | [Nn] [Oo]"#, "root")
                .unwrap(),
            LlamaSampler::temp(0.8),
            LlamaSampler::greedy(),
        ]);

        let token = sampler.sample(&context, batch.n_tokens() - 1).unwrap();

        assert!(
            !model.is_eog_token(token),
            "Grammar sampler should not allow EOS as first token"
        );

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .unwrap();
        let first_char = piece.chars().next().unwrap().to_lowercase().next().unwrap();

        assert!(
            first_char == 'y' || first_char == 'n',
            "Grammar should constrain first token to start with y/n, got: '{piece}'"
        );
    }

    #[test]
    #[serial]
    fn json_schema_grammar_sampler_constrains_output_to_json() {
        use crate::sampling::LlamaSampler;
        use std::sync::Arc;

        let (backend, model) = test_model::load_default_model().unwrap();

        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let prompt = "<|im_start|>user\nWhat is 2+2? Respond with a JSON object.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();
        let mut batch = crate::llama_batch::LlamaBatch::new(512, 1).unwrap();

        batch.add_sequence(&tokens, 0, false).unwrap();

        context.decode(&mut batch).unwrap();

        let grammar_str = crate::json_schema_to_grammar(
            r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#
        ).unwrap();

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::grammar(&model, &grammar_str, "root").unwrap(),
            LlamaSampler::temp(0.8),
            LlamaSampler::greedy(),
        ]);

        let token = sampler.sample(&context, batch.n_tokens() - 1).unwrap();

        assert!(
            !model.is_eog_token(token),
            "Grammar sampler should not allow EOS as first token"
        );

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .unwrap();

        assert!(
            piece.starts_with('{'),
            "JSON schema grammar should constrain first token to start with '{{', got: '{piece}'"
        );
    }

    #[test]
    #[serial]
    fn sample_with_grammar_produces_constrained_output_in_loop() {
        use crate::sampling::LlamaSampler;
        use std::sync::Arc;

        let (backend, model) = test_model::load_default_model().unwrap();

        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let prompt = "<|im_start|>user\nIs the sky blue? yes or no<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();
        let mut batch = crate::llama_batch::LlamaBatch::new(512, 1).unwrap();

        batch.add_sequence(&tokens, 0, false).unwrap();

        context.decode(&mut batch).unwrap();

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::grammar(&model, r#"root ::= "yes" | "no""#, "root").unwrap(),
            LlamaSampler::temp(0.8),
            LlamaSampler::greedy(),
        ]);

        let mut generated = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut position = batch.n_tokens();

        for iteration in 0..10 {
            let token = sampler.sample(&context, -1).unwrap();
            let is_eog = model.is_eog_token(token);

            eprintln!("  iteration={iteration} token={} eog={is_eog}", token.0);

            if is_eog {
                break;
            }

            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .unwrap();

            eprintln!("  piece='{piece}'");

            generated.push_str(&piece);

            batch.clear();
            batch.add(token, position, &[0], true).unwrap();
            position += 1;

            context.decode(&mut batch).unwrap();
        }

        let lowercase = generated.to_lowercase();

        assert!(
            lowercase == "yes" || lowercase == "no",
            "Grammar loop should produce 'yes' or 'no', got: '{generated}'"
        );
    }

    #[test]
    #[serial]
    fn sample_without_grammar_produces_multiple_tokens() {
        use crate::sampling::LlamaSampler;
        use std::sync::Arc;

        let (backend, model) = test_model::load_default_model().unwrap();

        let ctx_params = crate::context::params::LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let prompt =
            "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();
        let mut batch = crate::llama_batch::LlamaBatch::new(512, 1).unwrap();

        batch.add_sequence(&tokens, 0, false).unwrap();

        context.decode(&mut batch).unwrap();

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::greedy()]);

        let mut token_count = 0;
        let mut position = batch.n_tokens();

        for _ in 0..5 {
            let token = sampler.sample(&context, -1).unwrap();

            if model.is_eog_token(token) {
                break;
            }

            token_count += 1;

            batch.clear();
            batch.add(token, position, &[0], true).unwrap();
            position += 1;

            context.decode(&mut batch).unwrap();
        }

        assert!(
            token_count > 0,
            "Should produce at least one token without grammar"
        );
    }
}
