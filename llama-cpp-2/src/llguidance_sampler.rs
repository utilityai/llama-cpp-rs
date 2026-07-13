//! Pure Rust llguidance sampler for constrained decoding.
//!
//! Implements a custom `llama_sampler` using the `llguidance` and `toktrie` Rust crates
//! to enforce grammar constraints (JSON schema, regex, Lark, etc.) during token sampling.

use std::ffi::c_void;
use std::sync::Arc;

use llguidance::Matcher;
use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

/// Build a [`toktrie::TokEnv`] from a [`LlamaModel`]'s vocabulary.
///
/// Use this to construct your own `llguidance::ParserFactory` (with any slice regexes,
/// inference capabilities, or other configuration llguidance supports) and, from it, a
/// `llguidance::Matcher` to convert into a [`LlamaSampler`] via `LlamaSampler::from`.
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
//
// `ctx` is a boxed `llguidance::Matcher` directly — it owns its own tokenizer environment
// and parser state internally, so no extra wrapper struct is needed.

unsafe extern "C" fn llg_name(
    _smpl: *const llama_cpp_sys_2::llama_sampler,
) -> *const std::os::raw::c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut llama_cpp_sys_2::llama_sampler,
    token: llama_cpp_sys_2::llama_token,
) {
    let matcher = unsafe { &mut *(*smpl).ctx.cast::<Matcher>() };
    let _ = matcher.consume_token(token.cast_unsigned());
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_sys_2::llama_sampler,
    cur_p: *mut llama_cpp_sys_2::llama_token_data_array,
) {
    let matcher = unsafe { &mut *(*smpl).ctx.cast::<Matcher>() };
    let cur_p = unsafe { &mut *cur_p };

    let Ok(mask) = matcher.compute_mask_or_eos() else {
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
    let matcher = unsafe { &mut *(*smpl).ctx.cast::<Matcher>() };
    let _ = matcher.reset();
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_sys_2::llama_sampler,
) -> *mut llama_cpp_sys_2::llama_sampler {
    let matcher = unsafe { &*(*smpl).ctx.cast::<Matcher>() };
    let new_matcher = Box::new(matcher.deep_clone());
    unsafe {
        llama_cpp_sys_2::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_matcher).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut llama_cpp_sys_2::llama_sampler) {
    let ctx_ptr = unsafe { (*smpl).ctx.cast::<Matcher>() };
    if !ctx_ptr.is_null() {
        drop(unsafe { Box::from_raw(ctx_ptr) });
    }
}

static mut LLG_SAMPLER_I: llama_cpp_sys_2::llama_sampler_i = llama_cpp_sys_2::llama_sampler_i {
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

/// Wraps an already-built [`llguidance::Matcher`] into a [`LlamaSampler`].
///
/// Build a [`TokEnv`] via [`llguidance_build_tok_env`], your own `llguidance::ParserFactory`
/// (with whatever slices, inference capabilities, or other llguidance configuration you
/// need), and parse your grammar into a `Matcher` — then convert it here. This only adapts
/// the `Matcher` into the `llama_sampler` vtable; it has no opinion on how the `Matcher` was
/// built.
impl From<Matcher> for LlamaSampler {
    fn from(matcher: Matcher) -> Self {
        let ctx = Box::new(matcher);
        let sampler = unsafe {
            llama_cpp_sys_2::llama_sampler_init(
                &raw mut LLG_SAMPLER_I,
                Box::into_raw(ctx).cast::<c_void>(),
            )
        };
        LlamaSampler { sampler }
    }
}
