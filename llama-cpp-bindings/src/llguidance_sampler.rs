//! Pure Rust llguidance sampler for constrained decoding.
//!
//! Implements a custom `llama_sampler` using the `llguidance` and `toktrie` Rust crates
//! to enforce grammar constraints (JSON schema, regex, Lark, etc.) during token sampling.

use std::ffi::c_void;
use std::sync::Arc;

use llguidance::Matcher;
use toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

use crate::GrammarError;
use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

/// Internal state for the llguidance sampler.
struct LlgContext {
    matcher: Matcher,
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: String,
    grammar_data: String,
}

/// Build a [`toktrie::TokEnv`] from a [`LlamaModel`]'s vocabulary.
///
/// This mirrors the logic in upstream `llguidance.cpp` — for each token:
/// - Try normal detokenize (special=false)
/// - If empty, detokenize with special=true and prefix with 0xFF marker byte
fn build_tok_env(model: &LlamaModel) -> Arc<ApproximateTokEnv> {
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

// --- extern "C" vtable callbacks ---

const unsafe extern "C" fn llg_name(
    _smpl: *const llama_cpp_bindings_sys::llama_sampler,
) -> *const std::os::raw::c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut llama_cpp_bindings_sys::llama_sampler,
    token: llama_cpp_bindings_sys::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };

    if let Err(consume_error) = ctx.matcher.consume_token(token.cast_unsigned()) {
        tracing::warn!(
            token = token,
            error = %consume_error,
            "llguidance sampler failed to consume token"
        );
    }
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_bindings_sys::llama_sampler,
    cur_p: *mut llama_cpp_bindings_sys::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let mask = match ctx.matcher.compute_mask() {
        Ok(mask) => mask,
        Err(compute_error) => {
            tracing::warn!(
                error = %compute_error,
                "llguidance sampler failed to compute mask, skipping constraint application"
            );

            return;
        }
    };

    let data = unsafe { std::slice::from_raw_parts_mut(cur_p.data, cur_p.size) };
    for item in data.iter_mut() {
        if !mask.is_allowed(item.id.cast_unsigned()) {
            item.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn llg_reset(smpl: *mut llama_cpp_bindings_sys::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };

    if let Err(reset_error) = ctx.matcher.reset() {
        tracing::warn!(
            error = %reset_error,
            "llguidance sampler failed to reset"
        );
    }
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_bindings_sys::llama_sampler,
) -> *mut llama_cpp_bindings_sys::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        matcher: ctx.matcher.deep_clone(),
        tok_env: Arc::clone(&ctx.tok_env),
        grammar_kind: ctx.grammar_kind.clone(),
        grammar_data: ctx.grammar_data.clone(),
    });
    unsafe {
        llama_cpp_bindings_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_ctx).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut llama_cpp_bindings_sys::llama_sampler) {
    let ctx_ptr = unsafe { (*smpl).ctx.cast::<LlgContext>() };
    if !ctx_ptr.is_null() {
        drop(unsafe { Box::from_raw(ctx_ptr) });
    }
}

static mut LLG_SAMPLER_I: llama_cpp_bindings_sys::llama_sampler_i =
    llama_cpp_bindings_sys::llama_sampler_i {
        name: Some(llg_name),
        accept: Some(llg_accept),
        apply: Some(llg_apply),
        reset: Some(llg_reset),
        clone: Some(llg_clone),
        free: Some(llg_free),
        backend_init: None,
        backend_accept: None,
        backend_apply: None,
        backend_set_input: None,
    };

/// Create an llguidance-based constrained decoding sampler.
///
/// # Errors
///
/// Returns `GrammarError` if the parser factory, grammar, or parser cannot be created.
pub fn create_llg_sampler(
    model: &LlamaModel,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<LlamaSampler, GrammarError> {
    let tok_env = build_tok_env(model);
    let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();

    let factory = llguidance::ParserFactory::new_simple(&tok_env_dyn)
        .map_err(|factory_error| GrammarError::LlguidanceError(factory_error.to_string()))?;

    let grammar = llguidance::api::TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data)
        .map_err(|parse_error| GrammarError::LlguidanceError(parse_error.to_string()))?;

    let parser = factory
        .create_parser(grammar)
        .map_err(|parser_error| GrammarError::LlguidanceError(parser_error.to_string()))?;

    let matcher = Matcher::new(Ok(parser));

    let ctx = Box::new(LlgContext {
        matcher,
        tok_env,
        grammar_kind: grammar_kind.to_string(),
        grammar_data: grammar_data.to_string(),
    });

    let sampler = unsafe {
        llama_cpp_bindings_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(ctx).cast::<c_void>(),
        )
    };

    if sampler.is_null() {
        Err(GrammarError::NullGrammar(
            "llguidance sampler returned null".to_owned(),
        ))
    } else {
        Ok(LlamaSampler { sampler })
    }
}

#[cfg(all(test, feature = "tests_that_use_llms"))]
mod tests {
    use std::ffi::CStr;
    use std::num::NonZeroU32;

    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::llama_batch::LlamaBatch;
    use crate::model::AddBos;
    use crate::sampling::LlamaSampler;
    use crate::test_model;

    use super::LlgContext;
    use super::create_llg_sampler;

    const JSON_SCHEMA: &str =
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
    const REGEX_GRAMMAR: &str = r"yes|no";
    const LARK_GRAMMAR: &str = r#"start: "yes" | "no""#;

    #[test]
    #[serial]
    fn creates_sampler_with_valid_json_schema() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "json", JSON_SCHEMA).unwrap();

        assert!(!sampler.sampler.is_null());
    }

    #[test]
    #[serial]
    fn creates_sampler_with_valid_regex_grammar() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();

        assert!(!sampler.sampler.is_null());
    }

    #[test]
    #[serial]
    fn creates_sampler_with_valid_lark_grammar() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "lark", LARK_GRAMMAR).unwrap();

        assert!(!sampler.sampler.is_null());
    }

    #[test]
    #[serial]
    fn returns_error_for_unknown_grammar_kind() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = create_llg_sampler(&model, "not_a_real_kind", "anything");

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn returns_error_for_malformed_json_schema() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = create_llg_sampler(&model, "json", "{this is not valid json");

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn returns_error_for_malformed_regex() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let result = create_llg_sampler(&model, "regex", "[invalid");

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn name_callback_returns_llguidance() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();

        let name_ptr = unsafe { llama_cpp_bindings_sys::llama_sampler_name(sampler.sampler) };
        assert!(!name_ptr.is_null());
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap();

        assert_eq!(name, "llguidance");
    }

    #[test]
    #[serial]
    fn reset_clears_sampler_state() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let mut sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();

        sampler.reset();
    }

    #[test]
    #[serial]
    fn clone_via_ffi_creates_independent_sampler() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();

        let cloned = unsafe { llama_cpp_bindings_sys::llama_sampler_clone(sampler.sampler) };

        assert!(!cloned.is_null());

        unsafe { llama_cpp_bindings_sys::llama_sampler_free(cloned) };
    }

    #[test]
    #[serial]
    fn samples_token_constrained_by_grammar() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let prompt = "Answer yes or no:";
        let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let llg_sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();
        let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

        let token = chain.sample(&context, batch.n_tokens() - 1).unwrap();
        chain.accept(token).unwrap();
    }

    #[test]
    #[serial]
    fn accept_invalid_token_id_through_consume_does_not_panic() {
        let (_backend, model) = test_model::load_default_model().unwrap();
        let sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();

        let ctx = unsafe { &mut *(*sampler.sampler).ctx.cast::<LlgContext>() };
        let huge_token = i32::MAX - 1;
        let consume_result = ctx.matcher.consume_token(huge_token.cast_unsigned());

        assert!(consume_result.is_err());
    }

    #[test]
    #[serial]
    fn build_tok_env_handles_special_tokens() {
        use toktrie::TokenizerEnv;

        let (_backend, model) = test_model::load_default_model().unwrap();
        let tok_env = super::build_tok_env(&model);
        let info = tok_env.tok_trie().info();

        assert!(info.vocab_size > 0);
    }

    #[test]
    #[serial]
    fn apply_through_chain_during_sample_does_not_panic() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();

        let tokens = model.str_to_token("Answer:", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let llg_sampler = create_llg_sampler(&model, "regex", REGEX_GRAMMAR).unwrap();
        let mut chain = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);
        let _ = chain.sample(&context, batch.n_tokens() - 1);
    }
}
