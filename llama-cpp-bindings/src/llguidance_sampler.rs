use std::ffi::c_void;
use std::sync::Arc;

use llama_cpp_error_recorder::RecordedError;
use llama_cpp_error_recorder::record;
use toktrie::ApproximateTokEnv;

use crate::GrammarError;
use crate::grammar_matcher::GrammarMatcher;
use crate::mask_outcome::MaskOutcome;
use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;

struct LlgContext {
    grammar: GrammarMatcher,
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: String,
    grammar_data: String,
}

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

    if let Err(grammar_error) = ctx.grammar.consume_token(token.cast_unsigned()) {
        record(RecordedError::new(grammar_error));
    }
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_bindings_sys::llama_sampler,
    cur_p: *mut llama_cpp_bindings_sys::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let mask = match ctx.grammar.compute_mask() {
        Ok(MaskOutcome::Constrained(mask)) => mask,
        Ok(MaskOutcome::GrammarComplete) => return,
        Err(grammar_error) => {
            record(RecordedError::new(grammar_error));

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

    if let Err(grammar_error) = ctx.grammar.reset() {
        record(RecordedError::new(grammar_error));
    }
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_bindings_sys::llama_sampler,
) -> *mut llama_cpp_bindings_sys::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        grammar: ctx.grammar.deep_clone(),
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

/// # Errors
///
/// Returns `GrammarError` if the parser factory, grammar, or parser cannot be created.
pub fn create_llg_sampler(
    model: &LlamaModel,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<LlamaSampler, GrammarError> {
    let tok_env = model.approximate_tok_env()?;
    let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();

    let factory = llguidance::ParserFactory::new_simple(&tok_env_dyn)
        .map_err(|factory_error| GrammarError::LlguidanceError(factory_error.to_string()))?;

    let grammar = llguidance::api::TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data)
        .map_err(|parse_error| GrammarError::LlguidanceError(parse_error.to_string()))?;

    let parser = factory
        .create_parser(grammar)
        .map_err(|parser_error| GrammarError::LlguidanceError(parser_error.to_string()))?;

    let ctx = Box::new(LlgContext {
        grammar: GrammarMatcher::new(parser),
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
        Err(GrammarError::LlguidanceSamplerUnavailable)
    } else {
        Ok(LlamaSampler { sampler })
    }
}
