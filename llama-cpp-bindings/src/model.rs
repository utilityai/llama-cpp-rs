//! A safe wrapper around `llama_model`.

pub mod add_bos;
pub mod llama_chat_message;
pub mod llama_chat_template;
pub mod llama_lora_adapter;
pub mod llama_split_mode_parse_error;
pub mod params;
pub mod rope_type;
pub mod split_mode;
pub mod vocab_type;
pub mod vocab_type_from_int_error;

use std::ffi::{CStr, CString, c_char};
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::OnceLock;

use toktrie::ApproximateTokEnv;
use toktrie::TokRxInfo;
use toktrie::TokTrie;

use llama_cpp_bindings_types::ParsedChatMessage;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ReasoningMarkers;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_bindings_types::ToolCallMarkers;

use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
use crate::llama_backend::LlamaBackend;
use crate::llama_token_attrs::LlamaTokenAttrs;
use crate::llama_token_attrs_from_int_error::LlamaTokenAttrsFromIntError;
use crate::raw_chat_message::RawChatMessage;
use crate::resolved_tool_call_markers::ResolvedToolCallMarkers;
use crate::sampled_token::SampledToken;
use crate::sampled_token_classifier::SampledTokenClassifier;
use crate::streaming_markers::StreamingMarkers;
use crate::token::LlamaToken;
use crate::tool_call_format;
use crate::tool_call_format::ToolCallFormatOutcome;
use crate::tool_call_template_overrides;
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaLoraAdapterInitError, LlamaModelLoadError,
    MarkerDetectionError, MetaValError, ParseChatMessageError, StringToTokenError,
    TokenToStringError,
};

pub use add_bos::AddBos;
pub use llama_chat_message::LlamaChatMessage;
pub use llama_chat_template::LlamaChatTemplate;
pub use llama_lora_adapter::LlamaLoraAdapter;
pub use rope_type::RopeType;
pub use vocab_type::VocabType;
pub use vocab_type_from_int_error::VocabTypeFromIntError;

use params::LlamaModelParams;

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

/// A safe wrapper around `llama_model`.
pub struct LlamaModel {
    /// Raw pointer to the underlying `llama_model`.
    pub model: NonNull<llama_cpp_bindings_sys::llama_model>,
    tok_env: OnceLock<Arc<ApproximateTokEnv>>,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
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

        let size = invoke_rs_tokenize(
            self.vocab_ptr(),
            c_string.as_ptr(),
            c_string_len,
            buffer
                .as_mut_ptr()
                .cast::<llama_cpp_bindings_sys::llama_token>(),
            buffer_capacity,
            add_bos,
        )?;

        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size)?);
            invoke_rs_tokenize(
                self.vocab_ptr(),
                c_string.as_ptr(),
                c_string_len,
                buffer
                    .as_mut_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                -size,
                add_bos,
            )?
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
    ) -> Result<LlamaTokenAttrs, LlamaTokenAttrsFromIntError> {
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
    pub fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        let mut buffer: Vec<u8> = vec![0u8; buffer_size];
        let buffer_len = c_int::try_from(buffer.len())?;
        let lstrip = lstrip.map_or(0, |strip_count| i32::from(strip_count.get()));
        let size = unsafe {
            llama_cpp_bindings_sys::llama_token_to_piece(
                self.vocab_ptr(),
                token.0,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer_len,
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
                let written = usize::try_from(size)?;
                buffer.truncate(written);

                Ok(buffer)
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
    pub fn vocab_type(&self) -> Result<VocabType, VocabTypeFromIntError> {
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

            Ok(LlamaChatTemplate(chat_template_cstr.to_owned()))
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
        let mut out_model: *mut llama_cpp_bindings_sys::llama_model = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_load_model_from_file(
                cstr.as_ptr(),
                params.params,
                &raw mut out_model,
                &raw mut out_error,
            )
        };
        match status {
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK => {
                let model = NonNull::new(out_model)
                    .ok_or(LlamaModelLoadError::VendoredReturnedNull)?;
                Ok(Self {
                    model,
                    tok_env: OnceLock::new(),
                })
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG => {
                Err(LlamaModelLoadError::NullPathArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG => {
                Err(LlamaModelLoadError::NullOutModelArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG => {
                Err(LlamaModelLoadError::NullOutErrorArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_RETURNED_NULL => {
                if path.exists() {
                    Err(LlamaModelLoadError::VendoredReturnedNull)
                } else {
                    Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()))
                }
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
                Err(LlamaModelLoadError::ErrorStringAllocationFailed)
            }
            llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION => {
                let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
                Err(LlamaModelLoadError::VendoredThrewCxxException { message })
            }
            other => unreachable!(
                "llama_rs_load_model_from_file returned unrecognized status {other}"
            ),
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

    /// Build a streaming [`SampledTokenClassifier`] for this model.
    ///
    /// At construction the bindings detect reasoning markers (via the
    /// autoparser, with a chunked-thinking fallback for templates that consume
    /// thoughts via content blocks), tool-call markers, and the trailing
    /// generation-prompt slice. The classifier then runs a state machine over
    /// the decoded token stream — no per-model branches.
    ///
    /// If the model has no usable chat template the classifier is built in a
    /// blind mode that classifies every token as
    /// [`SampledToken::Undeterminable`].
    pub fn sampled_token_classifier(&self) -> SampledTokenClassifier<'_> {
        let markers = match self.streaming_markers() {
            Ok(markers) => markers,
            Err(detection_error) => {
                log::warn!(
                    "streaming markers detection failed; classifier will run blind: {detection_error}",
                );
                StreamingMarkers::default()
            }
        };

        SampledTokenClassifier::new(self, markers)
    }

    /// Detect reasoning / tool-call markers (as token-ID sequences) and the
    /// trailing generation-prompt slice for this model's chat template. The
    /// returned `StreamingMarkers` carry tokenised markers — never raw strings
    /// — so the classifier matches by `LlamaToken` equality rather than text
    /// scanning.
    ///
    /// # Errors
    /// Returns [`MarkerDetectionError`] when any underlying FFI call fails.
    pub fn streaming_markers(&self) -> Result<StreamingMarkers, MarkerDetectionError> {
        let (reasoning_open_str, reasoning_close_str) =
            invoke_detect_reasoning_markers(self.model.as_ptr())?;

        let tool_call_haystack = invoke_compute_tool_call_haystack(self.model.as_ptr())?;

        let autoparser_pair = tool_call_haystack.as_deref().and_then(
            crate::extract_tool_call_markers_from_haystack::extract_tool_call_markers_from_haystack,
        );

        let (autoparser_open, autoparser_close) = match autoparser_pair {
            Some(crate::tool_call_marker_pair::ToolCallMarkerPair { open, close }) => {
                (Some(open), Some(close))
            }
            None => (None, None),
        };

        let resolved_tool_call_markers =
            self.resolve_tool_call_marker_strings(autoparser_open, autoparser_close);

        Ok(StreamingMarkers {
            reasoning_open: self.tokenize_marker(reasoning_open_str.as_deref()),
            reasoning_close: self.tokenize_marker(reasoning_close_str.as_deref()),
            tool_call_open: self.tokenize_marker(resolved_tool_call_markers.open.as_deref()),
            tool_call_close: self.tokenize_marker(resolved_tool_call_markers.close.as_deref()),
        })
    }

    /// When the autoparser-driven FFI returned no tool-call markers, consult the
    /// per-template override registry so wrapper-known templates (Gemma 4,
    /// Mistral 3, ...) still drive the classifier.
    fn resolve_tool_call_marker_strings(
        &self,
        autoparser_open: Option<String>,
        autoparser_close: Option<String>,
    ) -> ResolvedToolCallMarkers {
        if autoparser_open
            .as_deref()
            .is_some_and(|raw| !raw.trim().is_empty())
        {
            return ResolvedToolCallMarkers {
                open: autoparser_open,
                close: autoparser_close,
            };
        }
        let Some(markers) = self.tool_call_markers() else {
            return ResolvedToolCallMarkers {
                open: autoparser_open,
                close: autoparser_close,
            };
        };
        let close = if markers.close.is_empty() {
            None
        } else {
            Some(markers.close)
        };
        ResolvedToolCallMarkers {
            open: Some(markers.open),
            close,
        }
    }

    /// # Errors
    /// Returns [`MarkerDetectionError`] when the underlying FFI call fails.
    pub fn reasoning_markers(&self) -> Result<Option<ReasoningMarkers>, MarkerDetectionError> {
        let (open, close) = invoke_detect_reasoning_markers(self.model.as_ptr())?;

        match (open, close) {
            (Some(open), Some(close)) if !open.is_empty() && !close.is_empty() => {
                Ok(Some(ReasoningMarkers { open, close }))
            }
            _ => Ok(None),
        }
    }

    /// Returns the rich tool-call marker bundle (open / separator / close /
    /// optional value-quote pair) for this model's chat template, sourced from
    /// the wrapper's per-template override registry. Returns `None` when no
    /// registered override matches — callers in that case fall back to
    /// llama.cpp's autoparser via [`Self::parse_chat_message`].
    #[must_use]
    pub fn tool_call_markers(&self) -> Option<ToolCallMarkers> {
        let template = match self.chat_template(None) {
            Ok(template) => template,
            Err(error) => {
                log::debug!(
                    "tool-call markers unavailable: chat template missing or invalid: {error}",
                );
                return None;
            }
        };
        let template_str = match template.to_str() {
            Ok(template_str) => template_str,
            Err(error) => {
                log::debug!(
                    "tool-call markers unavailable: chat template is not valid UTF-8: {error}",
                );
                return None;
            }
        };
        tool_call_template_overrides::detect(template_str)
    }

    fn tokenize_marker(&self, marker: Option<&str>) -> Option<Vec<LlamaToken>> {
        let marker = marker?.trim();
        if marker.is_empty() {
            return None;
        }
        match self.str_to_token(marker, AddBos::Never) {
            Ok(tokens) if !tokens.is_empty() => Some(tokens),
            Ok(_) => None,
            Err(tokenize_error) => {
                log::debug!(
                    "marker {marker:?} failed to tokenise; classifier will ignore it: {tokenize_error}",
                );
                None
            }
        }
    }

    /// Parse the assistant's output text into structured content, reasoning,
    /// and tool calls.
    ///
    /// Two passes, in order:
    /// 1. Duck-type the wrapper-side parsers across every known shape
    ///    (Qwen XML, GLM key-value, Gemma paired-quote, Mistral bracketed-JSON).
    ///    First match wins. The shapes are ordered so that more restrictive
    ///    shapes run first, which keeps the duck-type pass safe for inputs
    ///    that share an open marker but differ in inner structure.
    /// 2. Delegate to llama.cpp's `common_chat_parse`. If it succeeds the
    ///    result is `Recognized`; if it throws `ParseException` the result is
    ///    `Unrecognized` with the raw input plus the FFI's diagnostic, so the
    ///    caller can pass the unstructured tokens to the client.
    ///
    /// Empty tool-call `id` fields are filled with `call_{index}` before
    /// returning, so callers always see well-formed identifiers.
    ///
    /// `tools_json` is a JSON-array string of OpenAI-style tool definitions
    /// (use `"[]"` when no tools are in scope). `is_partial` switches between
    /// mid-stream (lenient) and final (strict) parses for the FFI step.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when `tools_json` is not valid JSON,
    /// the FFI returns a non-OK status other than `ParseException`, or
    /// accessor strings are not valid UTF-8.
    pub fn parse_chat_message(
        &self,
        tools_json: &str,
        input: &str,
        is_partial: bool,
    ) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
        let tools_value: serde_json::Value =
            serde_json::from_str(tools_json).map_err(ParseChatMessageError::ToolsJsonInvalid)?;
        if !tools_value.is_array() {
            return Err(ParseChatMessageError::ToolsJsonNotArray);
        }

        let reasoning_markers = self.reasoning_markers().ok().flatten();

        for candidate in tool_call_template_overrides::known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(calls) =
                tool_call_format::try_parse(input, &candidate)
            {
                let split =
                    split_reasoning_prefix(input, reasoning_markers.as_ref(), &candidate.open);
                let mut parsed = ParsedChatMessage::new(split.content, split.reasoning, calls);
                synthesize_missing_tool_call_ids(&mut parsed.tool_calls);
                return Ok(ChatMessageParseOutcome::Recognized(parsed));
            }
        }

        match self.parse_chat_message_via_ffi(tools_json, input, is_partial) {
            Ok(mut parsed) => {
                synthesize_missing_tool_call_ids(&mut parsed.tool_calls);
                Ok(ChatMessageParseOutcome::Recognized(parsed))
            }
            Err(ParseChatMessageError::ParseException { message }) => {
                Ok(ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                    tools_json: tools_json.to_owned(),
                    text: input.to_owned(),
                    is_partial,
                    ffi_error_message: message,
                }))
            }
            Err(other) => Err(other),
        }
    }

    fn parse_chat_message_via_ffi(
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
                i32::from(is_partial),
                &raw mut handle,
                &raw mut out_error,
            )
        };

        let parsed = match status {
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_OK => {
                collect_parsed_chat_message(handle)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_MODEL_ARG => {
                Err(ParseChatMessageError::ParseNullModelArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG => {
                Err(ParseChatMessageError::ParseNullInputArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG => {
                Err(ParseChatMessageError::ParseNullOutHandleArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG => {
                Err(ParseChatMessageError::ParseNullOutErrorArg)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_CHAT_TEMPLATE => {
                Err(ParseChatMessageError::ParseModelHasNoChatTemplate)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_VOCAB => {
                Err(ParseChatMessageError::ParseModelHasNoVocab)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED => {
                Err(ParseChatMessageError::ParseErrorStringAllocationFailed)
            }
            llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION => {
                let message =
                    unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
                out_error = ptr::null_mut();
                Err(ParseChatMessageError::ParseException { message })
            }
            other => unreachable!("llama_rs_parse_chat_message returned unrecognized status {other}"),
        };

        let mut free_error: *mut c_char = ptr::null_mut();
        let free_status = unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle, &raw mut free_error)
        };
        match (parsed, free_status) {
            (Ok(value), llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
                Ok(value)
            }
            (Ok(_), llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
                let message = unsafe {
                    crate::ffi_error_reader::read_and_free_cpp_error(free_error)
                };
                Err(ParseChatMessageError::FreeDestructorThrewCxxException { message })
            }
            (Ok(_), llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
                Err(ParseChatMessageError::FreeErrorStringAllocationFailed)
            }
            (Ok(_), other) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(free_error) };
                unreachable!("llama_rs_parsed_chat_free returned unrecognized status {other}")
            }
            (Err(parse_err), _) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(free_error) };
                Err(parse_err)
            }
        }
    }

    /// Render the model's chat template with the autoparser's synthetic
    /// no-tools and with-tools inputs. Returns `(output_no_tools,
    /// output_with_tools)`. Either side can be empty when the template throws
    /// during rendering. Useful for debugging tool-call marker detection.
    ///
    /// # Errors
    ///
    /// Returns [`MarkerDetectionError`] when the C++ analyzer throws or the FFI
    /// returns a non-OK status.
    pub fn diagnose_tool_call_synthetic_renders(
        &self,
    ) -> Result<(String, String), MarkerDetectionError> {
        let (no_tools, with_tools) = invoke_diagnose_tool_call_synthetic_renders(self.model.as_ptr())?;

        Ok((no_tools.unwrap_or_default(), with_tools.unwrap_or_default()))
    }
}

impl LlamaModel {
    /// Returns a process-cached, approximate token environment built from this model's vocabulary.
    ///
    /// The first call iterates the full vocabulary and constructs the trie; subsequent calls
    /// return the cached `Arc` without further FFI work.
    pub fn approximate_tok_env(&self) -> Arc<ApproximateTokEnv> {
        Arc::clone(self.tok_env.get_or_init(|| build_approximate_tok_env(self)))
    }
}

fn build_approximate_tok_env(model: &LlamaModel) -> Arc<ApproximateTokEnv> {
    let n_vocab = model.n_vocab().cast_unsigned();
    let tok_eos = {
        let eot = unsafe { llama_cpp_bindings_sys::llama_vocab_eot(model.vocab_ptr()) };
        if eot == -1 {
            model.token_eos().0.cast_unsigned()
        } else {
            eot.cast_unsigned()
        }
    };
    let info = TokRxInfo::new(n_vocab, tok_eos);

    let mut words = Vec::with_capacity(n_vocab as usize);

    for token_id in 0..n_vocab.cast_signed() {
        let token = LlamaToken(token_id);
        let bytes = model
            .token_to_piece_bytes(token, 32, false, None)
            .unwrap_or_default();
        if bytes.is_empty() {
            let special_bytes = model
                .token_to_piece_bytes(token, 32, true, None)
                .unwrap_or_default();
            if special_bytes.is_empty() {
                words.push(vec![]);
            } else {
                let mut marked = Vec::with_capacity(special_bytes.len() + 1);
                marked.push(0xFF);
                marked.extend(special_bytes);
                words.push(marked);
            }
        } else {
            words.push(bytes);
        }
    }

    let trie = TokTrie::from(&info, &words);
    Arc::new(ApproximateTokEnv::new(trie))
}

fn collect_parsed_chat_message(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    if handle.is_null() {
        return Ok(ParsedChatMessage::default());
    }

    let content = read_parsed_chat_content(handle)?;
    let reasoning_content = read_parsed_chat_reasoning_content(handle)?;
    let count = read_parsed_chat_tool_call_count(handle)?;

    let mut tool_calls = Vec::with_capacity(count);
    for index in 0..count {
        let id = read_parsed_chat_tool_call_id(handle, index)?;
        let name = read_parsed_chat_tool_call_name(handle, index)?;
        let arguments_json = read_parsed_chat_tool_call_arguments(handle, index)?;

        let arguments = ToolCallArguments::from_string(arguments_json);
        tool_calls.push(ParsedToolCall::new(id, name, arguments));
    }

    Ok(ParsedChatMessage::new(
        content,
        reasoning_content,
        tool_calls,
    ))
}

fn read_parsed_chat_content(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_content(
            handle,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ContentNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_content reported null out_string while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ContentErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ContentThrewCxxException { message })
        }
        other => unreachable!("llama_rs_parsed_chat_content returned unrecognized status {other}"),
    }
}

fn read_parsed_chat_reasoning_content(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content(
            handle,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ReasoningContentNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_reasoning_content reported null out_string while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ReasoningContentErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ReasoningContentThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_reasoning_content returned unrecognized status {other}"
        ),
    }
}

fn read_parsed_chat_tool_call_count(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<usize, ParseChatMessageError> {
    let mut out_count: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count(
            handle,
            &raw mut out_count,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ToolCallCountNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_tool_call_count reported null out_count while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ToolCallCountErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ToolCallCountThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_count returned unrecognized status {other}"
        ),
    }
}

fn read_parsed_chat_tool_call_id(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ToolCallIdNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_tool_call_id reported null out_string while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallIdIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ToolCallIdErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ToolCallIdThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_id returned unrecognized status {other}"
        ),
    }
}

fn read_parsed_chat_tool_call_name(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ToolCallNameNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_tool_call_name reported null out_string while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallNameIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ToolCallNameErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ToolCallNameThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_name returned unrecognized status {other}"
        ),
    }
}

fn read_parsed_chat_tool_call_arguments(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG => {
            Err(ParseChatMessageError::ToolCallArgumentsNullHandleArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG => {
            unreachable!(
                "llama_rs_parsed_chat_tool_call_arguments reported null out_string while we passed a valid pointer"
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallArgumentsIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::ToolCallArgumentsErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::ToolCallArgumentsThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_arguments returned unrecognized status {other}"
        ),
    }
}

fn consume_accessor_string(ptr: *mut c_char) -> Result<String, ParseChatMessageError> {
    if ptr.is_null() {
        return Ok(String::new());
    }
    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec();
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(ptr) };
    Ok(String::from_utf8(bytes)?)
}

struct ReasoningSplit {
    reasoning: String,
    content: String,
}

fn split_reasoning_prefix(
    input: &str,
    reasoning_markers: Option<&ReasoningMarkers>,
    tool_call_open: &str,
) -> ReasoningSplit {
    let content_only = || ReasoningSplit {
        reasoning: String::new(),
        content: prefix_before(input, tool_call_open),
    };

    let Some(reasoning_markers) = reasoning_markers else {
        return content_only();
    };
    let Some(open_pos) = input.find(&reasoning_markers.open) else {
        return content_only();
    };

    let after_open = &input[open_pos + reasoning_markers.open.len()..];
    let Some(close_offset) = after_open.find(&reasoning_markers.close) else {
        return content_only();
    };

    let reasoning = after_open[..close_offset].to_owned();
    let after_close = &after_open[close_offset + reasoning_markers.close.len()..];

    ReasoningSplit {
        reasoning,
        content: prefix_before(after_close, tool_call_open),
    }
}

fn prefix_before(text: &str, marker: &str) -> String {
    text.find(marker)
        .map_or_else(|| text.to_owned(), |pos| text[..pos].to_owned())
}

fn synthesize_missing_tool_call_ids(tool_calls: &mut [ParsedToolCall]) {
    for (index, call) in tool_calls.iter_mut().enumerate() {
        if call.id.is_empty() {
            call.id = format!("call_{index}");
        }
    }
}

fn invoke_detect_reasoning_markers(
    model: *const llama_cpp_bindings_sys::llama_model,
) -> Result<(Option<String>, Option<String>), MarkerDetectionError> {
    let mut out_open: *mut c_char = ptr::null_mut();
    let mut out_close: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_detect_reasoning_markers(
            model,
            &raw mut out_open,
            &raw mut out_close,
            &raw mut out_error,
        )
    };

    let parsed = match status {
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_OK => {
            collect_optional_cstr_pair(out_open, out_close)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_NULL_MODEL_ARG => {
            Err(MarkerDetectionError::DetectReasoningMarkersNullModelArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_OPEN_ARG => {
            Err(MarkerDetectionError::DetectReasoningMarkersNullOutOpenArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_CLOSE_ARG => {
            Err(MarkerDetectionError::DetectReasoningMarkersNullOutCloseArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_ERROR_ARG => {
            Err(MarkerDetectionError::DetectReasoningMarkersNullOutErrorArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::DetectReasoningMarkersErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::DetectReasoningMarkersVendoredThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_detect_reasoning_markers returned unrecognized status {other}"
        ),
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_open) };
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_close) };
    if !matches!(
        parsed,
        Err(MarkerDetectionError::DetectReasoningMarkersVendoredThrewCxxException { .. })
    ) {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
    }

    parsed
}

fn invoke_compute_tool_call_haystack(
    model: *const llama_cpp_bindings_sys::llama_model,
) -> Result<Option<String>, MarkerDetectionError> {
    let mut out_haystack: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_compute_tool_call_haystack(
            model,
            &raw mut out_haystack,
            &raw mut out_error,
        )
    };

    let parsed = match status {
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK => {
            read_optional_owned_cstr(out_haystack)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_MODEL_ARG => {
            Err(MarkerDetectionError::ComputeToolCallHaystackNullModelArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_HAYSTACK_ARG => {
            Err(MarkerDetectionError::ComputeToolCallHaystackNullOutHaystackArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_ERROR_ARG => {
            Err(MarkerDetectionError::ComputeToolCallHaystackNullOutErrorArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::ComputeToolCallHaystackErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::ComputeToolCallHaystackVendoredThrewCxxException { message })
        }
        other => unreachable!(
            "llama_rs_compute_tool_call_haystack returned unrecognized status {other}"
        ),
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_haystack) };
    if !matches!(
        parsed,
        Err(MarkerDetectionError::ComputeToolCallHaystackVendoredThrewCxxException { .. })
    ) {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
    }

    parsed
}

fn invoke_diagnose_tool_call_synthetic_renders(
    model: *const llama_cpp_bindings_sys::llama_model,
) -> Result<(Option<String>, Option<String>), MarkerDetectionError> {
    let mut out_no_tools: *mut c_char = ptr::null_mut();
    let mut out_with_tools: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_diagnose_tool_call_synthetic_renders(
            model,
            &raw mut out_no_tools,
            &raw mut out_with_tools,
            &raw mut out_error,
        )
    };

    let parsed = match status {
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK => {
            collect_optional_cstr_pair(out_no_tools, out_with_tools)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_MODEL_ARG => {
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersNullModelArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_NO_TOOLS_ARG => {
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersNullOutNoToolsArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_WITH_TOOLS_ARG => {
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersNullOutWithToolsArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_ERROR_ARG => {
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersNullOutErrorArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersVendoredThrewCxxException {
                message,
            })
        }
        other => unreachable!(
            "llama_rs_diagnose_tool_call_synthetic_renders returned unrecognized status {other}"
        ),
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_no_tools) };
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_with_tools) };
    if !matches!(
        parsed,
        Err(MarkerDetectionError::DiagnoseToolCallSyntheticRendersVendoredThrewCxxException { .. })
    ) {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
    }

    parsed
}

fn read_optional_owned_cstr(ptr: *const c_char) -> Result<Option<String>, MarkerDetectionError> {
    if ptr.is_null() {
        return Ok(None);
    }

    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec();

    Ok(Some(String::from_utf8(bytes)?))
}

fn invoke_rs_tokenize(
    vocab: *const llama_cpp_bindings_sys::llama_vocab,
    text: *const c_char,
    text_len: c_int,
    tokens: *mut llama_cpp_bindings_sys::llama_token,
    n_tokens_max: c_int,
    add_bos: bool,
) -> Result<c_int, StringToTokenError> {
    let mut out_count: i32 = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_tokenize(
            vocab,
            text,
            text_len,
            tokens,
            n_tokens_max,
            add_bos,
            true,
            &raw mut out_count,
            &raw mut out_error,
        )
    };
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG => {
            Err(StringToTokenError::NullVocabArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_TEXT_ARG => {
            Err(StringToTokenError::NullTextArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG => {
            Err(StringToTokenError::NullOutReturnedCountArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG => {
            Err(StringToTokenError::NullOutErrorArg)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(StringToTokenError::ErrorStringAllocationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(StringToTokenError::VendoredThrewCxxException { message })
        }
        other => unreachable!("llama_rs_tokenize returned unrecognized status {other}"),
    }
}

fn collect_optional_cstr_pair(
    first_ptr: *const c_char,
    second_ptr: *const c_char,
) -> Result<(Option<String>, Option<String>), MarkerDetectionError> {
    let first = read_optional_owned_cstr(first_ptr)?;
    let second = read_optional_owned_cstr(second_ptr)?;
    Ok((first, second))
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

