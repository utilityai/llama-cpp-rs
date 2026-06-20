//! Integration smoke test for MTP speculative decoding.
//!
//! Requires an MTP GGUF (embedded NextN head). Ignored by default; run with:
//! `LLAMA_MTP_MODEL=/path/to/model.gguf cargo test -p llama-cpp-2 --test mtp_speculative -- --ignored`
#![cfg(feature = "common")]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::speculative::MtpSpeculator;
use llama_cpp_2::token::LlamaToken;

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0 as i32
}

#[test]
#[ignore = "requires an MTP GGUF; set LLAMA_MTP_MODEL and run with --ignored"]
fn mtp_generates_and_drafts() {
    let model_path =
        std::env::var("LLAMA_MTP_MODEL").expect("set LLAMA_MTP_MODEL to an MTP GGUF path");

    let backend = LlamaBackend::init().unwrap();
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default()).unwrap();

    let n_draft = 4;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_n_rs_seq((n_draft + 4) as u32);

    let mut ctx_tgt = model.new_context(&backend, ctx_params.clone()).unwrap();
    let mut ctx_dft = model
        .new_mtp_context(&backend, &ctx_tgt, ctx_params)
        .unwrap();

    // SAFETY: both contexts outlive `spec` (all dropped at end of test).
    let mut spec =
        unsafe { MtpSpeculator::new(&ctx_tgt, &ctx_dft, n_draft, 0, 0.0, true) }.unwrap();
    assert!(spec.need_embd_nextn(), "MTP should need nextn embeddings");

    let tokens = model
        .str_to_token("The capital of France is", AddBos::Always)
        .unwrap();
    let mut batch = LlamaBatch::new(2048, 1);

    let head = &tokens[..tokens.len() - 1];
    for (i, t) in head.iter().enumerate() {
        batch.add(*t, i as i32, &[0], true).unwrap();
    }
    ctx_tgt.decode(&mut batch).unwrap();
    spec.process(&batch).unwrap();
    spec.begin(0, head).unwrap();

    let mut id_last = *tokens.last().unwrap();
    let mut n_past = (tokens.len() - 1) as i32;
    let mut prompt_tgt: Vec<LlamaToken> = head.to_vec();

    let mut generated = 0i32;
    let mut n_drafted = 0u64;
    let mut n_accept = 0u64;

    while generated < 16 {
        let draft = spec.draft(0, n_past, id_last, &prompt_tgt).unwrap();
        n_drafted += draft.len() as u64;

        ctx_dft
            .clear_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .unwrap();

        batch.clear();
        batch.add(id_last, n_past, &[0], true).unwrap();
        for (i, d) in draft.iter().enumerate() {
            batch.add(*d, n_past + 1 + i as i32, &[0], true).unwrap();
        }
        ctx_tgt.decode(&mut batch).unwrap();
        spec.process(&batch).unwrap();

        let mut ids = Vec::with_capacity(draft.len() + 1);
        let mut i = 0usize;
        loop {
            let t = argmax(ctx_tgt.get_logits_ith(i as i32));
            ids.push(t);
            if i == draft.len() || t != draft[i].0 {
                break;
            }
            i += 1;
        }
        let n_acc = (ids.len() - 1) as u16;
        n_accept += u64::from(n_acc);
        spec.accept(0, n_acc).unwrap();

        for &t in &ids {
            prompt_tgt.push(id_last);
            id_last = LlamaToken(t);
            n_past += 1;
            generated += 1;
            if generated >= 16 {
                break;
            }
        }

        ctx_tgt
            .clear_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .unwrap();
        ctx_dft
            .clear_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .unwrap();
    }

    eprintln!("generated={generated} n_drafted={n_drafted} n_accept={n_accept}");
    assert_eq!(generated, 16, "should generate the requested tokens");
    assert!(n_drafted > 0, "MTP head should produce draft tokens");
}
