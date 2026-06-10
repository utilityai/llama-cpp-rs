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

fn validate_string_length_for_tokenizer(length: usize) -> Result<c_int, StringToTokenError> {
    Ok(c_int::try_from(length)?)
}

fn cstring_with_validated_len(str: &str) -> Result<(CString, c_int), StringToTokenError> {
    let c_string = CString::new(str)?;
    let len = validate_string_length_for_tokenizer(c_string.as_bytes().len())?;
    Ok((c_string, len))
}

pub struct LlamaModel {
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

// SAFETY: `out_model` and `out_error` must be the pointers populated by the
// preceding `llama_rs_load_model_from_file` call (or null); `out_error` is read
// and freed only in the CXX-exception arm.
unsafe fn load_model_from_file_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_load_model_from_file_status,
    out_model: *mut llama_cpp_bindings_sys::llama_model,
    out_error: *mut c_char,
    path: &Path,
) -> Result<LlamaModel, LlamaModelLoadError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK => {
            let model = NonNull::new(out_model).ok_or(LlamaModelLoadError::Unloadable)?;
            Ok(LlamaModel {
                model,
                tok_env: OnceLock::new(),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_RETURNED_NULL => {
            if path.exists() {
                Err(LlamaModelLoadError::Unloadable)
            } else {
                Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()))
            }
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(LlamaModelLoadError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(LlamaModelLoadError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_load_model_from_file returned unrecognized status {other}")
        }
    }
}

// SAFETY: `handle` must be the parsed-chat handle (or null) and `out_error` must
// reference the pointer populated by the preceding `llama_rs_parse_chat_message`
// call. In the CXX-exception arm the error is read, freed, and the referenced
// pointer is nulled so the later free in the caller does not double-free.
unsafe fn parse_chat_message_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parse_chat_message_status,
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    out_error: *mut *mut c_char,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_OK => {
            collect_parsed_chat_message(handle)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_CHAT_TEMPLATE => {
            Err(ParseChatMessageError::NoChatTemplate)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_VOCAB => {
            Err(ParseChatMessageError::NoVocab)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(*out_error) };
            unsafe { *out_error = ptr::null_mut() };
            Err(ParseChatMessageError::ParseFailed { message })
        }
        other => {
            unreachable!("llama_rs_parse_chat_message returned unrecognized status {other}")
        }
    }
}

// SAFETY: `out_error` and `free_error` must be the pointers populated by the
// preceding parse and `llama_rs_parsed_chat_free` calls (or null); every arm
// frees each pointer exactly once across the two `llama_rs_string_free` calls.
unsafe fn parsed_chat_free_status_to_result(
    parsed: Result<ParsedChatMessage, ParseChatMessageError>,
    free_status: llama_cpp_bindings_sys::llama_rs_parsed_chat_free_status,
    out_error: *mut c_char,
    free_error: *mut c_char,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    match (parsed, free_status) {
        (Ok(value), llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK) => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Ok(value)
        }
        (
            Ok(_),
            llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
        ) => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(free_error) };
            Err(ParseChatMessageError::DestructorFailed { message })
        }
        (
            Ok(_),
            llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED,
        ) => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
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

fn reasoning_markers_from_marker_pair(
    open: Option<String>,
    close: Option<String>,
) -> Option<ReasoningMarkers> {
    match (open, close) {
        (Some(open), Some(close)) if !open.is_empty() && !close.is_empty() => {
            Some(ReasoningMarkers { open, close })
        }
        _ => None,
    }
}

fn outcome_from_via_ffi_result(
    via_ffi_result: Result<ParsedChatMessage, ParseChatMessageError>,
    tools_json: &str,
    input: &str,
    is_partial: bool,
) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
    match via_ffi_result {
        Ok(mut parsed) => {
            synthesize_missing_tool_call_ids(&mut parsed.tool_calls);
            Ok(ChatMessageParseOutcome::Recognized(parsed))
        }
        Err(ParseChatMessageError::ParseFailed { message }) => {
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

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_apply_chat_template` call (or null). The success arm reads
// and frees `out_string`; the CXX-exception arm reads and frees `out_error`.
unsafe fn apply_chat_template_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_apply_chat_template_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ApplyChatTemplateError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_OK => {
            Ok(unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_string) })
        }
        llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB => {
            Err(ApplyChatTemplateError::NoVocab)
        }
        llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_TEMPLATE_APPLICATION_FAILED => {
            Err(ApplyChatTemplateError::TemplateApplicationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ApplyChatTemplateError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ApplyChatTemplateError::Reported { message })
        }
        other => {
            unreachable!("llama_rs_apply_chat_template returned unrecognized status {other}")
        }
    }
}

impl LlamaModel {
    #[must_use]
    pub fn vocab_ptr(&self) -> *const llama_cpp_bindings_sys::llama_vocab {
        unsafe { llama_cpp_bindings_sys::llama_model_get_vocab(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the value returned by llama.cpp does not fit into a `u32`.
    pub fn n_ctx_train(&self) -> Result<u32, std::num::TryFromIntError> {
        let n_ctx_train = unsafe { llama_cpp_bindings_sys::llama_n_ctx_train(self.model.as_ptr()) };

        u32::try_from(n_ctx_train)
    }

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

    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_bos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_eos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_token_nl(self.vocab_ptr()) };
        LlamaToken(token)
    }

    #[must_use]
    pub fn is_eog_token(&self, token: &SampledToken) -> bool {
        let (SampledToken::Content(LlamaToken(id))
        | SampledToken::Reasoning(LlamaToken(id))
        | SampledToken::ToolCall(LlamaToken(id))
        | SampledToken::Undeterminable(LlamaToken(id))) = *token;

        unsafe { llama_cpp_bindings_sys::llama_token_is_eog(self.vocab_ptr(), id) }
    }

    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_bindings_sys::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    #[must_use]
    pub fn token_sep(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_bindings_sys::llama_vocab_sep(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// # Errors
    ///
    /// - if [`str`] contains a null byte
    /// - if an integer conversion fails during tokenization
    ///
    ///
    /// ```no_run
    /// use llama_cpp_bindings::model::LlamaModel;
    ///
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
        let (c_string, c_string_len) = cstring_with_validated_len(str)?;
        let vocab = self.vocab_ptr();

        tokenize_into_buffer(tokens_estimation, |tokens, n_tokens_max| {
            invoke_rs_tokenize(
                vocab,
                c_string.as_ptr(),
                c_string_len,
                tokens,
                n_tokens_max,
                add_bos,
            )
        })
    }

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

    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_n_vocab(self.vocab_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp emits a vocab type that is not known to this library.
    pub fn vocab_type(&self) -> Result<VocabType, VocabTypeFromIntError> {
        let vocab_type = unsafe { llama_cpp_bindings_sys::llama_vocab_type(self.vocab_ptr()) };

        VocabType::try_from(vocab_type)
    }

    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_n_embd(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_size(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_params(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the layer count returned by llama.cpp does not fit into a `u32`.
    pub fn n_layer(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_layer(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the KV head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head_kv(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head_kv(self.model.as_ptr()) })
    }

    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_hybrid(self.model.as_ptr()) }
    }

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

    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_model_meta_count(self.model.as_ptr()) }
    }

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

    #[must_use]
    pub fn rope_type(&self) -> Option<RopeType> {
        let raw = unsafe { llama_cpp_bindings_sys::llama_model_rope_type(self.model.as_ptr()) };

        rope_type::rope_type_from_raw(raw)
    }

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
        unsafe { load_model_from_file_status_to_result(status, out_model, out_error, path) }
    }

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
            return Err(LlamaLoraAdapterInitError::Unloadable);
        };

        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// # Errors
    /// Returns [`ApplyChatTemplateError`] if the model has no vocab, the template
    /// renders an empty prompt or cannot be rendered, or the renderer throws.
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        let roles: Vec<*const c_char> = chat
            .iter()
            .map(|chat_message| chat_message.role.as_ptr())
            .collect();
        let contents: Vec<*const c_char> = chat
            .iter()
            .map(|chat_message| chat_message.content.as_ptr())
            .collect();

        let mut out_string: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_apply_chat_template(
                self.model.as_ptr(),
                tmpl.0.as_ptr(),
                roles.as_ptr(),
                contents.as_ptr(),
                chat.len(),
                i32::from(add_ass),
                &raw mut out_string,
                &raw mut out_error,
            )
        };

        unsafe { apply_chat_template_status_to_result(status, out_string, out_error) }
    }

    /// # Errors
    /// Returns [`MarkerDetectionError`] when streaming-marker detection fails.
    /// The classifier is never constructed in a degraded "blind" state — a
    /// detection failure is surfaced to the caller instead of silently ignored.
    pub fn sampled_token_classifier(
        &self,
    ) -> Result<SampledTokenClassifier<'_>, MarkerDetectionError> {
        let markers = self.streaming_markers()?;

        Ok(SampledTokenClassifier::new(self, markers))
    }

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
            self.resolve_tool_call_marker_strings(autoparser_open, autoparser_close)?;

        Ok(StreamingMarkers {
            reasoning_open: self.tokenize_marker(reasoning_open_str.as_deref())?,
            reasoning_close: self.tokenize_marker(reasoning_close_str.as_deref())?,
            tool_call_open: self.tokenize_marker(resolved_tool_call_markers.open.as_deref())?,
            tool_call_close: self.tokenize_marker(resolved_tool_call_markers.close.as_deref())?,
        })
    }

    fn resolve_tool_call_marker_strings(
        &self,
        autoparser_open: Option<String>,
        autoparser_close: Option<String>,
    ) -> Result<ResolvedToolCallMarkers, MarkerDetectionError> {
        if autoparser_open
            .as_deref()
            .is_some_and(|raw| !raw.trim().is_empty())
        {
            return Ok(ResolvedToolCallMarkers {
                open: autoparser_open,
                close: autoparser_close,
            });
        }
        let Some(markers) = self.tool_call_markers()? else {
            return Ok(ResolvedToolCallMarkers {
                open: autoparser_open,
                close: autoparser_close,
            });
        };
        let close = if markers.close.is_empty() {
            None
        } else {
            Some(markers.close)
        };
        Ok(ResolvedToolCallMarkers {
            open: Some(markers.open),
            close,
        })
    }

    /// # Errors
    /// Returns [`MarkerDetectionError`] when the underlying FFI call fails.
    pub fn reasoning_markers(&self) -> Result<Option<ReasoningMarkers>, MarkerDetectionError> {
        let (open, close) = invoke_detect_reasoning_markers(self.model.as_ptr())?;

        Ok(reasoning_markers_from_marker_pair(open, close))
    }

    /// # Errors
    /// Returns [`MarkerDetectionError::ToolCallTemplateNotUtf8`] when the model
    /// has a chat template that is not valid UTF-8. A model with no chat
    /// template legitimately yields `Ok(None)`.
    pub fn tool_call_markers(&self) -> Result<Option<ToolCallMarkers>, MarkerDetectionError> {
        let template = match self.chat_template(None) {
            Ok(template) => template,
            Err(ChatTemplateError::MissingTemplate) => return Ok(None),
            Err(other) => return Err(MarkerDetectionError::ChatTemplateUnavailable(other)),
        };
        let template_str = template.to_str()?;

        Ok(tool_call_template_overrides::detect(template_str))
    }

    /// # Errors
    /// Returns [`StringToTokenError`] when a present, non-empty marker string
    /// fails to tokenise.
    fn tokenize_marker(
        &self,
        marker: Option<&str>,
    ) -> Result<Option<Vec<LlamaToken>>, StringToTokenError> {
        let Some(marker) = marker else {
            return Ok(None);
        };
        let marker = marker.trim();
        if marker.is_empty() {
            return Ok(None);
        }
        let tokens = self.str_to_token(marker, AddBos::Never)?;
        if tokens.is_empty() {
            Ok(None)
        } else {
            Ok(Some(tokens))
        }
    }

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

        let reasoning_markers = self.reasoning_markers()?;

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

        let via_ffi_result = self.parse_chat_message_via_ffi(tools_json, input, is_partial);

        outcome_from_via_ffi_result(via_ffi_result, tools_json, input, is_partial)
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

        let parsed =
            unsafe { parse_chat_message_status_to_result(status, handle, &raw mut out_error) };

        let mut free_error: *mut c_char = ptr::null_mut();
        let free_status = unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle, &raw mut free_error)
        };
        unsafe { parsed_chat_free_status_to_result(parsed, free_status, out_error, free_error) }
    }

    /// # Errors
    ///
    /// Returns [`MarkerDetectionError`] when the C++ analyzer throws or the FFI
    /// returns a non-OK status.
    pub fn diagnose_tool_call_synthetic_renders(
        &self,
    ) -> Result<(String, String), MarkerDetectionError> {
        let (no_tools, with_tools) =
            invoke_diagnose_tool_call_synthetic_renders(self.model.as_ptr())?;

        Ok((no_tools.unwrap_or_default(), with_tools.unwrap_or_default()))
    }
}

impl LlamaModel {
    /// # Errors
    /// Returns [`TokenToStringError`] when a token's byte piece cannot be
    /// retrieved. The legitimate "this token has no byte piece" case is treated
    /// as empty (not an error); a piece that overflows the probe buffer is
    /// re-read at the exact size rather than dropped.
    pub fn approximate_tok_env(&self) -> Result<Arc<ApproximateTokEnv>, TokenToStringError> {
        if let Some(env) = self.tok_env.get() {
            return Ok(Arc::clone(env));
        }
        let env = build_approximate_tok_env(self)?;
        Ok(Arc::clone(self.tok_env.get_or_init(|| env)))
    }
}

const TOK_ENV_PIECE_PROBE_SIZE: usize = 32;

fn token_piece_bytes_for_tok_env(
    model: &LlamaModel,
    token: LlamaToken,
    special: bool,
) -> Result<Vec<u8>, TokenToStringError> {
    match model.token_to_piece_bytes(token, TOK_ENV_PIECE_PROBE_SIZE, special, None) {
        Ok(bytes) => Ok(bytes),
        Err(TokenToStringError::UnknownTokenType) => Ok(Vec::new()),
        Err(TokenToStringError::InsufficientBufferSpace(required)) => {
            model.token_to_piece_bytes(token, required.unsigned_abs() as usize, special, None)
        }
        Err(other) => Err(other),
    }
}

fn build_approximate_tok_env(
    model: &LlamaModel,
) -> Result<Arc<ApproximateTokEnv>, TokenToStringError> {
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
        let bytes = token_piece_bytes_for_tok_env(model, token, false)?;
        if bytes.is_empty() {
            let special_bytes = token_piece_bytes_for_tok_env(model, token, true)?;
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
    Ok(Arc::new(ApproximateTokEnv::new(trie)))
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

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_parsed_chat_content` call (or null when no value/error
// was produced); each is read and freed in exactly one match arm.
unsafe fn parsed_chat_content_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_content_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!("llama_rs_parsed_chat_content returned unrecognized status {other}"),
    }
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
    unsafe { parsed_chat_content_status_to_result(status, out_string, out_error) }
}

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_parsed_chat_reasoning_content` call (or null when no
// value/error was produced); each is read and freed in exactly one match arm.
unsafe fn parsed_chat_reasoning_content_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_reasoning_content returned unrecognized status {other}"
        ),
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
    unsafe { parsed_chat_reasoning_content_status_to_result(status, out_string, out_error) }
}

// SAFETY: `out_error` must be the pointer populated by the preceding
// `llama_rs_parsed_chat_tool_call_count` call (or null when no error was
// produced); it is freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_count_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count_status,
    out_count: usize,
    out_error: *mut c_char,
) -> Result<usize, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_count returned unrecognized status {other}"
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
    unsafe { parsed_chat_tool_call_count_status_to_result(status, out_count, out_error) }
}

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_parsed_chat_tool_call_id` call (or null when no
// value/error was produced); each is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_id_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallIdIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_id returned unrecognized status {other}"
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
    unsafe { parsed_chat_tool_call_id_status_to_result(status, index, out_string, out_error) }
}

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_parsed_chat_tool_call_name` call (or null when no
// value/error was produced); each is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_name_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallNameIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_name returned unrecognized status {other}"
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
    unsafe { parsed_chat_tool_call_name_status_to_result(status, index, out_string, out_error) }
}

// SAFETY: `out_string` and `out_error` must be the pointers populated by the
// preceding `llama_rs_parsed_chat_tool_call_arguments` call (or null when no
// value/error was produced); each is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_arguments_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK => {
            consume_accessor_string(out_string)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallArgumentsIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(ParseChatMessageError::Reported { message })
        }
        other => unreachable!(
            "llama_rs_parsed_chat_tool_call_arguments returned unrecognized status {other}"
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
    unsafe {
        parsed_chat_tool_call_arguments_status_to_result(status, index, out_string, out_error)
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

// SAFETY: `out_open`, `out_close`, and `out_error` must be the pointers
// populated by the preceding `llama_rs_detect_reasoning_markers` call (or null).
// `out_open`/`out_close` are read but not freed here; `out_error` is freed only
// in the CXX-exception arm, mirroring the conditional cleanup in the caller.
unsafe fn detect_reasoning_markers_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_detect_reasoning_markers_status,
    out_open: *const c_char,
    out_close: *const c_char,
    out_error: *mut c_char,
) -> Result<(Option<String>, Option<String>), MarkerDetectionError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_OK => {
            collect_optional_cstr_pair(out_open, out_close)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::ReasoningMarkerDetectionFailed { message })
        }
        other => unreachable!(
            "llama_rs_detect_reasoning_markers returned unrecognized status {other}"
        ),
    }
}

const fn cxx_exception_owns_out_error<TValue>(
    parsed: &Result<TValue, MarkerDetectionError>,
) -> bool {
    matches!(
        parsed,
        Err(MarkerDetectionError::ReasoningMarkerDetectionFailed { .. }
            | MarkerDetectionError::ToolCallHaystackComputationFailed { .. }
            | MarkerDetectionError::ToolCallSyntheticRenderDiagnosisFailed { .. })
    )
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

    let parsed = unsafe {
        detect_reasoning_markers_status_to_result(status, out_open, out_close, out_error)
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_open) };
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_close) };
    if !cxx_exception_owns_out_error(&parsed) {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
    }

    parsed
}

// SAFETY: `out_haystack` and `out_error` must be the pointers populated by the
// preceding `llama_rs_compute_tool_call_haystack` call (or null). `out_haystack`
// is read but not freed here; `out_error` is freed only in the CXX-exception
// arm, mirroring the conditional cleanup in the caller.
unsafe fn compute_tool_call_haystack_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_compute_tool_call_haystack_status,
    out_haystack: *const c_char,
    out_error: *mut c_char,
) -> Result<Option<String>, MarkerDetectionError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK => {
            read_optional_owned_cstr(out_haystack)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::ToolCallHaystackComputationFailed { message })
        }
        other => unreachable!(
            "llama_rs_compute_tool_call_haystack returned unrecognized status {other}"
        ),
    }
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

    let parsed =
        unsafe { compute_tool_call_haystack_status_to_result(status, out_haystack, out_error) };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_haystack) };
    if !cxx_exception_owns_out_error(&parsed) {
        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
    }

    parsed
}

// SAFETY: `out_no_tools`, `out_with_tools`, and `out_error` must be the pointers
// populated by the preceding `llama_rs_diagnose_tool_call_synthetic_renders`
// call (or null). The render pointers are read but not freed here; `out_error`
// is freed only in the CXX-exception arm, mirroring the cleanup in the caller.
unsafe fn diagnose_tool_call_synthetic_renders_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_diagnose_tool_call_synthetic_renders_status,
    out_no_tools: *const c_char,
    out_with_tools: *const c_char,
    out_error: *mut c_char,
) -> Result<(Option<String>, Option<String>), MarkerDetectionError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK => {
            collect_optional_cstr_pair(out_no_tools, out_with_tools)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MarkerDetectionError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(MarkerDetectionError::ToolCallSyntheticRenderDiagnosisFailed { message })
        }
        other => unreachable!(
            "llama_rs_diagnose_tool_call_synthetic_renders returned unrecognized status {other}"
        ),
    }
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

    let parsed = unsafe {
        diagnose_tool_call_synthetic_renders_status_to_result(
            status,
            out_no_tools,
            out_with_tools,
            out_error,
        )
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_no_tools) };
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_with_tools) };
    if !cxx_exception_owns_out_error(&parsed) {
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

// SAFETY: `out_error` must be the pointer populated by the preceding
// `llama_rs_tokenize` call (or null when no error was produced); it is read and
// freed only in the CXX-exception arm.
unsafe fn tokenize_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_tokenize_status,
    out_count: c_int,
    out_error: *mut c_char,
) -> Result<c_int, StringToTokenError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(StringToTokenError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { crate::ffi_error_reader::read_and_free_cpp_error(out_error) };
            Err(StringToTokenError::Reported { message })
        }
        other => unreachable!("llama_rs_tokenize returned unrecognized status {other}"),
    }
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
    unsafe { tokenize_status_to_result(status, out_count, out_error) }
}

fn checked_token_buffer_capacity(capacity: usize) -> Result<c_int, StringToTokenError> {
    Ok(c_int::try_from(capacity)?)
}

fn checked_token_count(size: i32) -> Result<usize, StringToTokenError> {
    Ok(usize::try_from(size)?)
}

fn tokenize_into_buffer(
    estimated_capacity: usize,
    invoke: impl Fn(
        *mut llama_cpp_bindings_sys::llama_token,
        c_int,
    ) -> Result<c_int, StringToTokenError>,
) -> Result<Vec<LlamaToken>, StringToTokenError> {
    let mut buffer: Vec<LlamaToken> = Vec::with_capacity(estimated_capacity);
    let buffer_capacity = checked_token_buffer_capacity(buffer.capacity())?;

    let size = invoke(
        buffer
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>(),
        buffer_capacity,
    )?;

    let size = if size.is_negative() {
        buffer.reserve_exact(checked_token_count(-size)?);
        invoke(
            buffer
                .as_mut_ptr()
                .cast::<llama_cpp_bindings_sys::llama_token>(),
            -size,
        )?
    } else {
        size
    };

    let size = checked_token_count(size)?;

    // SAFETY: `size` <= `capacity` and llama-cpp has initialized elements up to `size`
    unsafe { buffer.set_len(size) }

    Ok(buffer)
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
    fn checked_token_buffer_capacity_overflow_returns_error() {
        assert!(super::checked_token_buffer_capacity(usize::MAX).is_err());
    }

    #[test]
    fn checked_token_buffer_capacity_in_range_returns_value() {
        assert_eq!(super::checked_token_buffer_capacity(8), Ok(8));
    }

    #[test]
    fn checked_token_count_negative_returns_error() {
        assert!(super::checked_token_count(-1).is_err());
    }

    #[test]
    fn checked_token_count_non_negative_returns_value() {
        assert_eq!(super::checked_token_count(5), Ok(5));
    }

    #[test]
    fn tokenize_into_buffer_single_pass_sets_length() {
        let buffer = super::tokenize_into_buffer(8, |_tokens, _n_tokens_max| Ok(3)).unwrap();

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn tokenize_into_buffer_grows_buffer_when_first_pass_reports_negative_size() {
        let call_count = std::cell::Cell::new(0);
        let buffer = super::tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 { Ok(-20) } else { Ok(15) }
        })
        .unwrap();

        assert_eq!(buffer.len(), 15);
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn tokenize_into_buffer_propagates_invocation_error() {
        let result = super::tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            Err(crate::StringToTokenError::NotEnoughMemory)
        });

        assert_eq!(result, Err(crate::StringToTokenError::NotEnoughMemory));
    }

    #[test]
    fn tokenize_into_buffer_propagates_second_invocation_error() {
        let call_count = std::cell::Cell::new(0);
        let result = super::tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 {
                Ok(-20)
            } else {
                Err(crate::StringToTokenError::NotEnoughMemory)
            }
        });

        assert_eq!(result, Err(crate::StringToTokenError::NotEnoughMemory));
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn tokenize_into_buffer_negative_final_size_returns_conversion_error() {
        let call_count = std::cell::Cell::new(0);
        let result = super::tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 { Ok(-20) } else { Ok(-5) }
        });

        assert_eq!(
            result.unwrap_err(),
            crate::StringToTokenError::CIntConversionError(usize::try_from(-5i32).unwrap_err())
        );
    }

    #[test]
    fn read_optional_owned_cstr_invalid_utf8_returns_error() {
        let invalid_utf8_with_terminator: [u8; 3] = [0xFF, 0xFE, 0x00];
        let result = super::read_optional_owned_cstr(
            invalid_utf8_with_terminator
                .as_ptr()
                .cast::<std::ffi::c_char>(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::MarkerDetectionError::MarkerUtf8Error(
                String::from_utf8(vec![0xFF, 0xFE]).unwrap_err()
            )
        );
    }

    #[test]
    fn collect_optional_cstr_pair_first_invalid_utf8_returns_error() {
        let invalid_utf8_with_terminator: [u8; 3] = [0xFF, 0xFE, 0x00];
        let valid_with_terminator: [u8; 3] = [b'o', b'k', 0x00];
        let result = super::collect_optional_cstr_pair(
            invalid_utf8_with_terminator
                .as_ptr()
                .cast::<std::ffi::c_char>(),
            valid_with_terminator.as_ptr().cast::<std::ffi::c_char>(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::MarkerDetectionError::MarkerUtf8Error(
                String::from_utf8(vec![0xFF, 0xFE]).unwrap_err()
            )
        );
    }

    #[test]
    fn collect_optional_cstr_pair_second_invalid_utf8_returns_error() {
        let valid_with_terminator: [u8; 3] = [b'o', b'k', 0x00];
        let invalid_utf8_with_terminator: [u8; 3] = [0xFF, 0xFE, 0x00];
        let result = super::collect_optional_cstr_pair(
            valid_with_terminator.as_ptr().cast::<std::ffi::c_char>(),
            invalid_utf8_with_terminator
                .as_ptr()
                .cast::<std::ffi::c_char>(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::MarkerDetectionError::MarkerUtf8Error(
                String::from_utf8(vec![0xFF, 0xFE]).unwrap_err()
            )
        );
    }
}

#[cfg(test)]
mod ffi_status_mapping_tests {
    use std::ffi::c_char;
    use std::mem::discriminant;
    use std::path::Path;
    use std::ptr;

    use llama_cpp_bindings_types::ParsedChatMessage;
    use llama_cpp_bindings_types::ParsedToolCall;
    use llama_cpp_bindings_types::ReasoningMarkers;
    use llama_cpp_bindings_types::ToolCallArguments;

    use super::ReasoningSplit;
    use super::compute_tool_call_haystack_status_to_result;
    use super::cxx_exception_owns_out_error;
    use super::detect_reasoning_markers_status_to_result;
    use super::diagnose_tool_call_synthetic_renders_status_to_result;
    use super::load_model_from_file_status_to_result;
    use super::outcome_from_via_ffi_result;
    use super::parse_chat_message_status_to_result;
    use super::parsed_chat_content_status_to_result;
    use super::parsed_chat_free_status_to_result;
    use super::parsed_chat_reasoning_content_status_to_result;
    use super::parsed_chat_tool_call_arguments_status_to_result;
    use super::parsed_chat_tool_call_count_status_to_result;
    use super::parsed_chat_tool_call_id_status_to_result;
    use super::parsed_chat_tool_call_name_status_to_result;
    use super::reasoning_markers_from_marker_pair;
    use super::split_reasoning_prefix;
    use super::tokenize_status_to_result;
    use crate::ChatMessageParseOutcome;
    use crate::LlamaModelLoadError;
    use crate::MarkerDetectionError;
    use crate::ParseChatMessageError;
    use crate::RawChatMessage;
    use crate::StringToTokenError;

    #[test]
    fn cxx_exception_owns_out_error_classifies_each_failure_variant() {
        assert!(cxx_exception_owns_out_error::<()>(&Err(
            MarkerDetectionError::ReasoningMarkerDetectionFailed {
                message: String::new()
            }
        )));
        assert!(cxx_exception_owns_out_error::<()>(&Err(
            MarkerDetectionError::ToolCallHaystackComputationFailed {
                message: String::new()
            }
        )));
        assert!(cxx_exception_owns_out_error::<()>(&Err(
            MarkerDetectionError::ToolCallSyntheticRenderDiagnosisFailed {
                message: String::new()
            }
        )));
        assert!(!cxx_exception_owns_out_error::<()>(&Ok(())));
    }

    #[test]
    fn load_model_from_file_ok_with_null_model_is_unloadable() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(result.unwrap_err(), LlamaModelLoadError::Unloadable);
    }

    #[test]
    fn load_model_from_file_vendored_returned_null_for_missing_path_is_file_not_found() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/definitely/missing/model.gguf"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            LlamaModelLoadError::FileNotFound(
                Path::new("/definitely/missing/model.gguf").to_path_buf()
            )
        );
    }

    #[test]
    fn load_model_from_file_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(result.unwrap_err(), LlamaModelLoadError::NotEnoughMemory);
    }

    #[test]
    fn load_model_from_file_cxx_exception_is_reported() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            LlamaModelLoadError::Reported {
                message: "unknown error".to_owned()
            }
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_load_model_from_file returned unrecognized status")]
    fn load_model_from_file_unrecognized_status_panics() {
        let _ = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::llama_rs_load_model_from_file_status::MAX,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };
    }

    #[test]
    fn parse_chat_message_ok_with_null_handle_is_default_message() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_OK,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(result.unwrap(), ParsedChatMessage::default());
    }

    #[test]
    fn parse_chat_message_no_chat_template_maps_to_no_chat_template() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_CHAT_TEMPLATE,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NoChatTemplate)
        );
    }

    #[test]
    fn parse_chat_message_no_vocab_maps_to_no_vocab() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_VOCAB,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NoVocab)
        );
    }

    #[test]
    fn parse_chat_message_allocation_failed_is_not_enough_memory() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parse_chat_message_cxx_exception_is_parse_failed_and_nulls_error() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::ParseFailed {
                message: String::new()
            })
        );
        assert!(out_error.is_null());
    }

    #[test]
    #[should_panic(expected = "llama_rs_parse_chat_message returned unrecognized status")]
    fn parse_chat_message_unrecognized_status_panics() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let _ = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parse_chat_message_status::MAX,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };
    }

    #[test]
    fn parsed_chat_free_ok_returns_parsed_value() {
        let parsed = Ok(ParsedChatMessage::default());
        let result = unsafe {
            parsed_chat_free_status_to_result(
                parsed,
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), ParsedChatMessage::default());
    }

    #[test]
    fn parsed_chat_free_destructor_threw_is_destructor_failed() {
        let parsed = Ok(ParsedChatMessage::default());
        let result = unsafe {
            parsed_chat_free_status_to_result(
                parsed,
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::DestructorFailed {
                message: String::new()
            })
        );
    }

    #[test]
    fn parsed_chat_free_allocation_failed_is_not_enough_memory() {
        let parsed = Ok(ParsedChatMessage::default());
        let result = unsafe {
            parsed_chat_free_status_to_result(
                parsed,
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_free_propagates_existing_parse_error() {
        let parsed = Err(ParseChatMessageError::NoVocab);
        let result = unsafe {
            parsed_chat_free_status_to_result(
                parsed,
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NoVocab)
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_parsed_chat_free returned unrecognized status")]
    fn parsed_chat_free_unrecognized_status_panics() {
        let parsed = Ok(ParsedChatMessage::default());
        let _ = unsafe {
            parsed_chat_free_status_to_result(
                parsed,
                llama_cpp_bindings_sys::llama_rs_parsed_chat_free_status::MAX,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_content_ok_with_null_string_is_empty() {
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn parsed_chat_content_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_content_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_parsed_chat_content returned unrecognized status")]
    fn parsed_chat_content_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_content_status::MAX,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_reasoning_content_ok_with_null_string_is_empty() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn parsed_chat_reasoning_content_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_reasoning_content_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(
        expected = "llama_rs_parsed_chat_reasoning_content returned unrecognized status"
    )]
    fn parsed_chat_reasoning_content_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content_status::MAX,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_tool_call_count_ok_returns_count() {
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK,
                7,
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn parsed_chat_tool_call_count_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_count_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_parsed_chat_tool_call_count returned unrecognized status")]
    fn parsed_chat_tool_call_count_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count_status::MAX,
                0,
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_tool_call_id_ok_with_null_string_is_empty() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn parsed_chat_tool_call_id_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS,
                4,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallIdIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallIdIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 4);
    }

    #[test]
    fn parsed_chat_tool_call_id_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_id_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_parsed_chat_tool_call_id returned unrecognized status")]
    fn parsed_chat_tool_call_id_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id_status::MAX,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_tool_call_name_ok_with_null_string_is_empty() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn parsed_chat_tool_call_name_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS,
                2,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallNameIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallNameIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 2);
    }

    #[test]
    fn parsed_chat_tool_call_name_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_name_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_parsed_chat_tool_call_name returned unrecognized status")]
    fn parsed_chat_tool_call_name_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name_status::MAX,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn parsed_chat_tool_call_arguments_ok_with_null_string_is_empty() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn parsed_chat_tool_call_arguments_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS,
                9,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallArgumentsIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallArgumentsIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 9);
    }

    #[test]
    fn parsed_chat_tool_call_arguments_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_arguments_cxx_exception_is_reported() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::Reported {
                message: String::new()
            })
        );
    }

    #[test]
    #[should_panic(
        expected = "llama_rs_parsed_chat_tool_call_arguments returned unrecognized status"
    )]
    fn parsed_chat_tool_call_arguments_unrecognized_status_panics() {
        let _ = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments_status::MAX,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn detect_reasoning_markers_ok_with_null_pointers_is_none_pair() {
        let result = unsafe {
            detect_reasoning_markers_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_OK,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok((None, None)));
    }

    #[test]
    fn detect_reasoning_markers_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            detect_reasoning_markers_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MarkerDetectionError::NotEnoughMemory));
    }

    #[test]
    fn detect_reasoning_markers_cxx_exception_is_detection_failed() {
        let result = unsafe {
            detect_reasoning_markers_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(MarkerDetectionError::ReasoningMarkerDetectionFailed {
                message: "unknown error".to_owned()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_detect_reasoning_markers returned unrecognized status")]
    fn detect_reasoning_markers_unrecognized_status_panics() {
        let _ = unsafe {
            detect_reasoning_markers_status_to_result(
                llama_cpp_bindings_sys::llama_rs_detect_reasoning_markers_status::MAX,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn compute_tool_call_haystack_ok_with_null_pointer_is_none() {
        let result = unsafe {
            compute_tool_call_haystack_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(None));
    }

    #[test]
    fn compute_tool_call_haystack_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            compute_tool_call_haystack_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MarkerDetectionError::NotEnoughMemory));
    }

    #[test]
    fn compute_tool_call_haystack_cxx_exception_is_computation_failed() {
        let result = unsafe {
            compute_tool_call_haystack_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(MarkerDetectionError::ToolCallHaystackComputationFailed {
                message: "unknown error".to_owned()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_compute_tool_call_haystack returned unrecognized status")]
    fn compute_tool_call_haystack_unrecognized_status_panics() {
        let _ = unsafe {
            compute_tool_call_haystack_status_to_result(
                llama_cpp_bindings_sys::llama_rs_compute_tool_call_haystack_status::MAX,
                ptr::null(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn diagnose_tool_call_synthetic_renders_ok_with_null_pointers_is_none_pair() {
        let result = unsafe {
            diagnose_tool_call_synthetic_renders_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok((None, None)));
    }

    #[test]
    fn diagnose_tool_call_synthetic_renders_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            diagnose_tool_call_synthetic_renders_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MarkerDetectionError::NotEnoughMemory));
    }

    #[test]
    fn diagnose_tool_call_synthetic_renders_cxx_exception_is_diagnosis_failed() {
        let result = unsafe {
            diagnose_tool_call_synthetic_renders_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(
                MarkerDetectionError::ToolCallSyntheticRenderDiagnosisFailed {
                    message: "unknown error".to_owned()
                }
            )
        );
    }

    #[test]
    #[should_panic(
        expected = "llama_rs_diagnose_tool_call_synthetic_renders returned unrecognized status"
    )]
    fn diagnose_tool_call_synthetic_renders_unrecognized_status_panics() {
        let _ = unsafe {
            diagnose_tool_call_synthetic_renders_status_to_result(
                llama_cpp_bindings_sys::llama_rs_diagnose_tool_call_synthetic_renders_status::MAX,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn tokenize_ok_returns_count() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK,
                12,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(12));
    }

    #[test]
    fn tokenize_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(StringToTokenError::NotEnoughMemory));
    }

    #[test]
    fn tokenize_cxx_exception_is_reported() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(StringToTokenError::Reported {
                message: "unknown error".to_owned()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_tokenize returned unrecognized status")]
    fn tokenize_unrecognized_status_panics() {
        let _ = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::llama_rs_tokenize_status::MAX,
                0,
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn apply_chat_template_ok_returns_rendered_prompt() {
        unsafe extern "C" {
            fn strdup(text: *const c_char) -> *mut c_char;
        }
        let rendered = std::ffi::CString::new("<bos>rendered prompt").unwrap();
        let out_string = unsafe { strdup(rendered.as_ptr()) };
        let result = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_OK,
                out_string,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok("<bos>rendered prompt".to_owned()));
    }

    #[test]
    fn apply_chat_template_no_vocab_maps_to_no_vocab() {
        let result = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(crate::ApplyChatTemplateError::NoVocab));
    }

    #[test]
    fn apply_chat_template_application_failed_maps_to_template_application_failed() {
        let result = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_TEMPLATE_APPLICATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(crate::ApplyChatTemplateError::TemplateApplicationFailed)
        );
    }

    #[test]
    fn apply_chat_template_allocation_failed_maps_to_not_enough_memory() {
        let result = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(crate::ApplyChatTemplateError::NotEnoughMemory));
    }

    #[test]
    fn apply_chat_template_cxx_exception_is_reported() {
        unsafe extern "C" {
            fn strdup(text: *const c_char) -> *mut c_char;
        }
        let message = std::ffi::CString::new("renderer exploded").unwrap();
        let out_error = unsafe { strdup(message.as_ptr()) };
        let result = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_APPLY_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(crate::ApplyChatTemplateError::Reported {
                message: "renderer exploded".to_owned()
            })
        );
    }

    #[test]
    #[should_panic(expected = "llama_rs_apply_chat_template returned unrecognized status")]
    fn apply_chat_template_unrecognized_status_panics() {
        let _ = unsafe {
            super::apply_chat_template_status_to_result(
                llama_cpp_bindings_sys::llama_rs_apply_chat_template_status::MAX,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
    }

    #[test]
    fn split_reasoning_prefix_without_markers_returns_content_up_to_tool_call_open() {
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("answer<tool>rest", None, "<tool>");

        assert!(reasoning.is_empty());
        assert_eq!(content, "answer");
    }

    #[test]
    fn split_reasoning_prefix_with_missing_open_marker_returns_content_only() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            close: "</think>".to_owned(),
        };
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("plain answer", Some(&markers), "<tool>");

        assert!(reasoning.is_empty());
        assert_eq!(content, "plain answer");
    }

    #[test]
    fn split_reasoning_prefix_with_missing_close_marker_returns_content_only() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            close: "</think>".to_owned(),
        };
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("<think>unterminated", Some(&markers), "<tool>");

        assert!(reasoning.is_empty());
        assert_eq!(content, "<think>unterminated");
    }

    #[test]
    fn split_reasoning_prefix_extracts_reasoning_and_trailing_content() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            close: "</think>".to_owned(),
        };
        let ReasoningSplit { reasoning, content } = split_reasoning_prefix(
            "<think>deduce</think>answer<tool>tail",
            Some(&markers),
            "<tool>",
        );

        assert_eq!(reasoning, "deduce");
        assert_eq!(content, "answer");
    }

    #[test]
    fn reasoning_markers_from_marker_pair_with_both_present_builds_markers() {
        let markers = reasoning_markers_from_marker_pair(
            Some("<think>".to_owned()),
            Some("</think>".to_owned()),
        );

        assert_eq!(
            markers,
            Some(ReasoningMarkers {
                open: "<think>".to_owned(),
                close: "</think>".to_owned()
            })
        );
    }

    #[test]
    fn reasoning_markers_from_marker_pair_with_empty_marker_is_none() {
        let markers =
            reasoning_markers_from_marker_pair(Some(String::new()), Some("</think>".to_owned()));

        assert!(markers.is_none());
    }

    #[test]
    fn reasoning_markers_from_marker_pair_with_missing_marker_is_none() {
        let markers = reasoning_markers_from_marker_pair(None, Some("</think>".to_owned()));

        assert!(markers.is_none());
    }

    #[test]
    fn outcome_from_via_ffi_result_recognized_synthesizes_tool_call_ids() {
        let parsed = ParsedChatMessage::new(
            "answer".to_owned(),
            String::new(),
            vec![ParsedToolCall::new(
                String::new(),
                "tool".to_owned(),
                ToolCallArguments::default(),
            )],
        );

        let outcome = outcome_from_via_ffi_result(Ok(parsed), "[]", "answer", false);

        assert_eq!(
            outcome.unwrap(),
            ChatMessageParseOutcome::Recognized(ParsedChatMessage::new(
                "answer".to_owned(),
                String::new(),
                vec![ParsedToolCall::new(
                    "call_0".to_owned(),
                    "tool".to_owned(),
                    ToolCallArguments::default(),
                )],
            ))
        );
    }

    #[test]
    fn outcome_from_via_ffi_result_parse_failed_is_unrecognized_with_raw_message() {
        let outcome = outcome_from_via_ffi_result(
            Err(ParseChatMessageError::ParseFailed {
                message: "boom".to_owned(),
            }),
            "[]",
            "garbled",
            true,
        );

        assert_eq!(
            outcome.unwrap(),
            ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                tools_json: "[]".to_owned(),
                text: "garbled".to_owned(),
                is_partial: true,
                ffi_error_message: "boom".to_owned(),
            })
        );
    }

    #[test]
    fn outcome_from_via_ffi_result_other_error_propagates() {
        let outcome =
            outcome_from_via_ffi_result(Err(ParseChatMessageError::NoVocab), "[]", "x", false);

        assert_eq!(
            discriminant(&outcome.unwrap_err()),
            discriminant(&ParseChatMessageError::NoVocab)
        );
    }
}
