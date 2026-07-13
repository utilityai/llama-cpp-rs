//! Pure Rust llguidance sampler for constrained decoding.
//!
//! Implements a custom `llama_sampler` using the `llguidance` and `toktrie` Rust crates
//! to enforce grammar constraints (JSON schema, regex, Lark, etc.) during token sampling.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use llguidance::Matcher;
use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::GrammarError;

/// Internal state for the llguidance sampler.
///
/// A [`Matcher`] is self-contained (it owns its own tokenizer environment and parser
/// state internally), so this is all the per-instance state the vtable callbacks need.
struct LlgContext {
    matcher: Matcher,
}

/// Build a [`toktrie::TokEnv`] from a [`LlamaModel`]'s vocabulary.
///
/// Use this to construct your own `llguidance::ParserFactory` (with any slice regexes,
/// inference capabilities, or other configuration llguidance supports) and, from it, a
/// `llguidance::Matcher` to pass to [`create_llg_sampler_from_matcher`].
///
/// This mirrors the logic in upstream `llguidance.cpp` — for each token:
/// - Try normal detokenize (special=false)
/// - If empty, detokenize with special=true and prefix with 0xFF marker byte
pub fn llguidance_build_tok_env(model: &LlamaModel) -> TokEnv {
    let n_vocab = model.n_vocab().cast_unsigned();
    let tok_eos = {
        let eot = unsafe { llama_cpp_sys_2::llama_vocab_eot(model.vocab_ptr()) };
        if eot == -1 {
            model.token_eos().0.cast_unsigned()
        } else {
            eot.cast_unsigned()
        }
    };
    let info = TokRxInfo::new(n_vocab, tok_eos);

    let mut words = Vec::with_capacity(n_vocab as usize);
    for i in 0..n_vocab.cast_signed() {
        let token = LlamaToken(i);
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

unsafe extern "C" fn llg_name(
    _smpl: *const llama_cpp_sys_2::llama_sampler,
) -> *const std::os::raw::c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut llama_cpp_sys_2::llama_sampler,
    token: llama_cpp_sys_2::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let _ = ctx.matcher.consume_token(token.cast_unsigned());
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_sys_2::llama_sampler,
    cur_p: *mut llama_cpp_sys_2::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let Ok(mask) = ctx.matcher.compute_mask_or_eos() else {
        return;
    };

    let data = unsafe { std::slice::from_raw_parts_mut(cur_p.data, cur_p.size) };
    for item in data.iter_mut() {
        if !mask.is_allowed(item.id.cast_unsigned()) {
            item.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn llg_reset(smpl: *mut llama_cpp_sys_2::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let _ = ctx.matcher.reset();
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_sys_2::llama_sampler,
) -> *mut llama_cpp_sys_2::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        matcher: ctx.matcher.deep_clone(),
    });
    unsafe {
        llama_cpp_sys_2::llama_sampler_init(llg_vtable(), Box::into_raw(new_ctx).cast::<c_void>())
    }
}

unsafe extern "C" fn llg_free(smpl: *mut llama_cpp_sys_2::llama_sampler) {
    let ctx_ptr = unsafe { (*smpl).ctx.cast::<LlgContext>() };
    if !ctx_ptr.is_null() {
        drop(unsafe { Box::from_raw(ctx_ptr) });
    }
}

/// Returns a pointer to the llguidance sampler vtable.
///
/// The vtable's function pointers never change after construction, so a single instance
/// is built lazily and shared by every llguidance sampler (and its clones) — only the
/// per-instance `ctx` differs. This avoids a `static mut`.
fn llg_vtable() -> *mut llama_cpp_sys_2::llama_sampler_i {
    static VTABLE: OnceLock<Box<llama_cpp_sys_2::llama_sampler_i>> = OnceLock::new();
    let vtable = VTABLE.get_or_init(|| {
        Box::new(llama_cpp_sys_2::llama_sampler_i {
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
        })
    });
    std::ptr::from_ref(vtable.as_ref()).cast_mut()
}

/// Wrap an already-built [`llguidance::Matcher`] into a [`LlamaSampler`].
///
/// This is the generic entry point: build a [`TokEnv`] via [`llguidance_build_tok_env`], your
/// own `llguidance::ParserFactory` (with whatever slices, inference capabilities, or other
/// llguidance configuration you need), parse your grammar into a `Matcher`, and hand it
/// here. This function only adapts the `Matcher` into the `llama_sampler` vtable — it has
/// no opinion on how the `Matcher` was built.
pub(crate) fn create_llg_sampler_from_matcher(matcher: Matcher) -> LlamaSampler {
    let ctx = Box::new(LlgContext { matcher });
    let sampler = unsafe {
        llama_cpp_sys_2::llama_sampler_init(llg_vtable(), Box::into_raw(ctx).cast::<c_void>())
    };
    LlamaSampler { sampler }
}

/// Create an llguidance-based constrained decoding sampler.
///
/// Uses `ParserFactory::new_simple`, which pre-builds a `SlicedBiasComputer` with the
/// four general JSON regexes. For custom slice regexes or other `ParserFactory`
/// configuration, build your own `llguidance::Matcher` (via [`llguidance_build_tok_env`] and the
/// `llguidance` crate directly) and pass it to [`create_llg_sampler_from_matcher`] instead.
pub(crate) fn create_llg_sampler(
    model: &LlamaModel,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<LlamaSampler, GrammarError> {
    let tok_env = llguidance_build_tok_env(model);
    let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();

    let factory = llguidance::ParserFactory::new_simple(&tok_env_dyn)
        .map_err(|_| GrammarError::NullGrammar)?;

    let grammar = llguidance::api::TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data)
        .map_err(|_| GrammarError::NullGrammar)?;

    let parser = factory
        .create_parser(grammar)
        .map_err(|_| GrammarError::NullGrammar)?;

    let matcher = Matcher::new(Ok(parser));

    let ctx = Box::new(LlgContext { matcher });

    let sampler = unsafe {
        llama_cpp_sys_2::llama_sampler_init(llg_vtable(), Box::into_raw(ctx).cast::<c_void>())
    };

    if sampler.is_null() {
        Err(GrammarError::NullGrammar)
    } else {
        Ok(LlamaSampler { sampler })
    }
}
