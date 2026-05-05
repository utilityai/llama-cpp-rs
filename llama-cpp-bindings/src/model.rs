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
use crate::ffi_status_to_i32::status_to_i32;
use crate::llama_backend::LlamaBackend;
use crate::parsed_chat_message::ParsedChatMessage;
use crate::parsed_tool_call::ParsedToolCall;
use crate::sampled_token::SampledToken;
use crate::sampled_token_classifier::SampledTokenClassifier;
use crate::sampled_token_classifier::SampledTokenClassifierMarkers;
use crate::sampled_token_classifier::TokenBoundary;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, MetaValError, ParseChatMessageError, ReasoningClassifierError,
    StringToTokenError, TokenToStringError,
};

pub mod add_bos;
pub mod llama_chat_message;
pub mod llama_chat_template;
pub mod llama_lora_adapter;
pub mod params;
pub mod rope_type;
pub mod split_mode;
pub mod vocab_type;

pub use add_bos::AddBos;
pub use llama_chat_message::LlamaChatMessage;
pub use llama_chat_template::LlamaChatTemplate;
pub use llama_lora_adapter::LlamaLoraAdapter;
pub use rope_type::RopeType;
pub use vocab_type::{LlamaTokenTypeFromIntError, VocabType};

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
                    self.token_to_piece(
                        &SampledToken::Content(llama_token),
                        &mut decoder,
                        decode_special,
                        None,
                    ),
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
    pub fn is_eog_token(&self, token: &SampledToken) -> bool {
        let (SampledToken::Content(LlamaToken(id))
        | SampledToken::Reasoning(LlamaToken(id))
        | SampledToken::ToolCall(LlamaToken(id))
        | SampledToken::Undeterminable(LlamaToken(id))) = *token;

        unsafe { llama_cpp_bindings_sys::llama_token_is_eog(self.vocab_ptr(), id) }
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
        token: &SampledToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError> {
        let (SampledToken::Content(inner)
        | SampledToken::Reasoning(inner)
        | SampledToken::ToolCall(inner)
        | SampledToken::Undeterminable(inner)) = *token;
        let bytes = match self.token_to_piece_bytes(inner, 8, special, lstrip) {
            Err(TokenToStringError::InsufficientBufferSpace(required_size)) => {
                let buffer_size: usize = (-required_size).try_into()?;

                self.token_to_piece_bytes(inner, buffer_size, special, lstrip)
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

    /// Build a [`SampledTokenClassifier`] for this model by detecting both the
    /// reasoning and tool-call section markers via llama.cpp's chat-template
    /// analyzer and resolving each pair to single Control-attribute token ids.
    ///
    /// Either marker pair (or both) may be absent — the resulting classifier
    /// reports tokens as `Content` outside any block, `Reasoning`/`ToolCall`
    /// inside the corresponding block, or `Undeterminable` when neither pair
    /// is known.
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningClassifierError`] when the C++ analyzer throws, when a
    /// detected marker does not tokenize to exactly one token, or when the resolved
    /// token does not have the [`LlamaTokenAttr::Control`] attribute.
    pub fn sampled_token_classifier(
        &self,
    ) -> Result<SampledTokenClassifier, ReasoningClassifierError> {
        let reasoning = self.detect_marker_strings(
            llama_cpp_bindings_sys::llama_rs_detect_reasoning_markers,
        )?;
        let tool_call = self.detect_marker_strings(
            llama_cpp_bindings_sys::llama_rs_detect_tool_call_markers,
        )?;

        Ok(SampledTokenClassifier::new(SampledTokenClassifierMarkers {
            reasoning: self.resolve_optional_boundary(reasoning)?,
            tool_call: self.resolve_optional_boundary(tool_call)?,
        }))
    }

    /// Render the chat template with the autoparser's standard tool-call
    /// synthetic inputs. Returns `(output_no_tools, output_with_tools)`. Each
    /// Parse the assistant's output text via llama.cpp's `common_chat_parse`,
    /// driven by the model's autoparser-built peg parser. Returns structured
    /// content / reasoning / tool-call data — never a raw JSON blob to
    /// deserialize on the Rust side.
    ///
    /// `tools_json` is a JSON-array string of OpenAI-style tool definitions
    /// (use `"[]"` when no tools are in scope). `is_partial` switches between
    /// mid-stream (lenient) and final (strict) parses.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when the FFI returns a non-OK
    /// status, the C++ side throws, or accessor strings are not valid UTF-8.
    pub fn parse_chat_message(
        &self,
        tools_json: &str,
        input: &str,
        is_partial: bool,
    ) -> Result<ParsedChatMessage, ParseChatMessageError> {
        let tools_cstring = CString::new(tools_json)
            .map_err(|err| ParseChatMessageError::ToolsSerialization(err.to_string()))?;
        let input_cstring = CString::new(input)
            .map_err(|err| ParseChatMessageError::ToolsSerialization(err.to_string()))?;

        let mut handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_parse_chat_message(
                self.model.as_ptr(),
                tools_cstring.as_ptr(),
                input_cstring.as_ptr(),
                if is_partial { 1 } else { 0 },
                &raw mut handle,
                &raw mut out_error,
            )
        };

        let parsed = match status {
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK => collect_parsed_chat_message(handle),
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_EXCEPTION => {
                let message = read_optional_owned_cstr_lossy(out_error);
                Err(ParseChatMessageError::ParseException(message))
            }
            other => Err(ParseChatMessageError::FfiError(status_to_i32(other))),
        };

        unsafe { llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle) };
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };

        parsed
    }

    /// can be empty when the template throws during rendering. Useful for
    /// debugging tool-call marker detection.
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningClassifierError`] when the C++ analyzer throws or
    /// the FFI returns a non-OK status.
    pub fn diagnose_tool_call_synthetic_renders(
        &self,
    ) -> Result<(String, String), ReasoningClassifierError> {
        let mut out_no_tools: *mut c_char = ptr::null_mut();
        let mut out_with_tools: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_diagnose_tool_call_synthetic_renders(
                self.model.as_ptr(),
                &raw mut out_no_tools,
                &raw mut out_with_tools,
                &raw mut out_error,
            )
        };

        let parsed = (|| match status {
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK => {
                let no_tools = read_optional_owned_cstr(out_no_tools)?;
                let with_tools = read_optional_owned_cstr(out_with_tools)?;

                Ok((no_tools.unwrap_or_default(), with_tools.unwrap_or_default()))
            }
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_EXCEPTION => {
                let message = read_optional_owned_cstr_lossy(out_error);

                Err(ReasoningClassifierError::AnalyzeException(message))
            }
            other => Err(ReasoningClassifierError::FfiError(status_to_i32(other))),
        })();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_no_tools) };
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_with_tools) };
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };

        parsed
    }

    fn detect_marker_strings(
        &self,
        detect_fn: unsafe extern "C" fn(
            *const llama_cpp_bindings_sys::llama_model,
            *mut *mut c_char,
            *mut *mut c_char,
            *mut *mut c_char,
        ) -> llama_cpp_bindings_sys::llama_rs_status,
    ) -> Result<(Option<String>, Option<String>), ReasoningClassifierError> {
        let mut out_open: *mut c_char = ptr::null_mut();
        let mut out_close: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            detect_fn(
                self.model.as_ptr(),
                &raw mut out_open,
                &raw mut out_close,
                &raw mut out_error,
            )
        };

        let parsed = (|| match status {
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK => {
                let open_string = read_optional_owned_cstr(out_open)?;
                let close_string = read_optional_owned_cstr(out_close)?;

                Ok((open_string, close_string))
            }
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_EXCEPTION => {
                let message = read_optional_owned_cstr_lossy(out_error);

                Err(ReasoningClassifierError::AnalyzeException(message))
            }
            other => Err(ReasoningClassifierError::FfiError(status_to_i32(other))),
        })();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_open) };
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_close) };
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };

        parsed
    }

    fn resolve_optional_boundary(
        &self,
        markers: (Option<String>, Option<String>),
    ) -> Result<Option<TokenBoundary>, ReasoningClassifierError> {
        let (Some(open_marker), Some(close_marker)) = markers else {
            return Ok(None);
        };

        let open = self.resolve_open_marker_token(open_marker.trim())?;
        let close = self.resolve_close_marker_token(close_marker.trim())?;

        Ok(Some(TokenBoundary { open, close }))
    }

    fn resolve_open_marker_token(
        &self,
        marker: &str,
    ) -> Result<LlamaToken, ReasoningClassifierError> {
        let tokens = self.str_to_token(marker, AddBos::Never)?;

        if tokens.len() != 1 {
            return Err(ReasoningClassifierError::OpenMarkerNotSingleToken {
                marker: marker.to_string(),
                token_count: tokens.len(),
            });
        }

        let token = tokens[0];
        let attrs = self.token_attr(token)?;

        if !is_special_marker_attr(attrs) {
            return Err(ReasoningClassifierError::OpenMarkerNotSpecial {
                marker: marker.to_string(),
            });
        }

        Ok(token)
    }

    fn resolve_close_marker_token(
        &self,
        marker: &str,
    ) -> Result<LlamaToken, ReasoningClassifierError> {
        let tokens = self.str_to_token(marker, AddBos::Never)?;

        if tokens.len() != 1 {
            return Err(ReasoningClassifierError::CloseMarkerNotSingleToken {
                marker: marker.to_string(),
                token_count: tokens.len(),
            });
        }

        let token = tokens[0];
        let attrs = self.token_attr(token)?;

        if !is_special_marker_attr(attrs) {
            return Err(ReasoningClassifierError::CloseMarkerNotSpecial {
                marker: marker.to_string(),
            });
        }

        Ok(token)
    }
}

fn is_special_marker_attr(attrs: LlamaTokenAttrs) -> bool {
    attrs.contains(LlamaTokenAttr::Control) || attrs.contains(LlamaTokenAttr::UserDefined)
}

fn collect_parsed_chat_message(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    if handle.is_null() {
        return Ok(ParsedChatMessage::default());
    }

    let content = read_owned_cstr_for_parse(unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_content(handle)
    })?;
    let reasoning_content = read_owned_cstr_for_parse(unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content(handle)
    })?;

    let count =
        unsafe { llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count(handle) };

    let mut tool_calls = Vec::with_capacity(count);
    for index in 0..count {
        let id = read_owned_cstr_for_parse(unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id(handle, index)
        })?;
        let name = read_owned_cstr_for_parse(unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name(handle, index)
        })?;
        let arguments_json = read_owned_cstr_for_parse(unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments(handle, index)
        })?;

        tool_calls.push(ParsedToolCall::new(id, name, arguments_json));
    }

    Ok(ParsedChatMessage::new(content, reasoning_content, tool_calls))
}

fn read_owned_cstr_for_parse(ptr: *mut c_char) -> Result<String, ParseChatMessageError> {
    if ptr.is_null() {
        return Ok(String::new());
    }

    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec();
    let owned = String::from_utf8(bytes)?;

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(ptr) };

    Ok(owned)
}

fn read_optional_owned_cstr(
    ptr: *const c_char,
) -> Result<Option<String>, ReasoningClassifierError> {
    if ptr.is_null() {
        return Ok(None);
    }

    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec();

    Ok(Some(String::from_utf8(bytes)?))
}

fn read_optional_owned_cstr_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
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
