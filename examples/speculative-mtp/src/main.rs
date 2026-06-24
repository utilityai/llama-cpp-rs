//! MTP (Multi-Token Prediction / `NextN`) speculative decoding example + benchmark.
//!
//! Uses a single GGUF whose embedded `NextN` head drafts tokens from the target model's own hidden
//! state. `--baseline` runs plain greedy decoding for a comparable tok/s.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::speculative::MtpSpeculator;
use llama_cpp_2::token::LlamaToken;

#[derive(Parser, Debug)]
#[command(about = "MTP speculative decoding example + benchmark")]
struct Args {
    /// Path to the MTP GGUF (with embedded `NextN` head).
    #[arg(long)]
    model: PathBuf,
    /// Prompt.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// Max tokens to generate.
    #[arg(long, default_value_t = 128)]
    n_predict: i32,
    /// Max draft tokens per speculative step.
    #[arg(long, default_value_t = 4)]
    n_draft: i32,
    /// Context size.
    #[arg(long, default_value_t = 4096)]
    n_ctx: u32,
    /// GPU layers to offload (0 = CPU only).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
    /// Recurrent-state rollback snapshots per seq (0 = auto = `n_draft` + 4).
    #[arg(long, default_value_t = 0)]
    n_rs_seq: u32,
    /// Run plain greedy decoding (no speculation) for a baseline tok/s.
    #[arg(long)]
    baseline: bool,
    /// Suppress generated text; print only the benchmark line.
    #[arg(long)]
    quiet: bool,
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

fn emit(model: &LlamaModel, decoder: &mut encoding_rs::Decoder, tok: LlamaToken, quiet: bool) {
    if quiet {
        return;
    }
    if let Ok(piece) = model.token_to_piece(tok, decoder, true, None) {
        print!("{piece}");
        std::io::stdout().flush().ok();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .context("failed to load model")?;

    let n_rs_seq = if args.n_rs_seq == 0 {
        (args.n_draft + 4) as u32
    } else {
        args.n_rs_seq
    };
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(args.n_ctx))
        .with_n_rs_seq(n_rs_seq);

    let tokens = model
        .str_to_token(&args.prompt, AddBos::Always)
        .context("failed to tokenize prompt")?;
    anyhow::ensure!(tokens.len() >= 2, "prompt must be at least 2 tokens");

    if args.baseline {
        run_baseline(&backend, &model, &ctx_params, &args, &tokens)
    } else {
        run_mtp(&backend, &model, &ctx_params, &args, &tokens)
    }
}

fn run_baseline(
    backend: &LlamaBackend,
    model: &LlamaModel,
    ctx_params: &LlamaContextParams,
    args: &Args,
    tokens: &[LlamaToken],
) -> Result<()> {
    let mut ctx = model
        .new_context(backend, ctx_params.clone())
        .context("failed to create context")?;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(args.n_ctx as usize, 1);

    for (i, t) in tokens.iter().enumerate() {
        batch.add(*t, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch).context("prefill decode failed")?;

    if !args.quiet {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }

    let mut n_past = tokens.len() as i32;
    let mut generated = 0i32;
    let t_start = Instant::now();

    while generated < args.n_predict {
        let tok = LlamaToken(argmax(ctx.get_logits_ith(batch.n_tokens() - 1)));
        if model.is_eog_token(tok) {
            break;
        }
        emit(model, &mut decoder, tok, args.quiet);
        generated += 1;
        batch.clear();
        batch.add(tok, n_past, &[0], true)?;
        n_past += 1;
        ctx.decode(&mut batch).context("decode failed")?;
    }

    // baseline does one target decode per token, so target_decodes == generated.
    report(
        "baseline",
        generated,
        generated as u64,
        generated as u64,
        generated as u64,
        t_start,
    );
    Ok(())
}

fn run_mtp(
    backend: &LlamaBackend,
    model: &LlamaModel,
    ctx_params: &LlamaContextParams,
    args: &Args,
    tokens: &[LlamaToken],
) -> Result<()> {
    let mut ctx_tgt = model
        .new_context(backend, ctx_params.clone())
        .context("failed to create target context")?;
    // SAFETY: ctx_tgt outlives ctx_dft — both live until this function returns, so the raw
    // `ctx_other` pointer the draft context holds into ctx_tgt stays valid.
    let mut ctx_dft = unsafe { model.new_mtp_context(backend, &ctx_tgt, ctx_params.clone()) }
        .context("failed to create MTP draft context")?;

    // SAFETY: ctx_tgt and ctx_dft outlive `spec` — all three live until this function returns.
    let mut spec = unsafe { MtpSpeculator::new(&ctx_tgt, &ctx_dft, args.n_draft, 0, 0.0, true) }
        .context("failed to init MTP speculator")?;
    anyhow::ensure!(
        spec.need_embd_nextn(),
        "MTP speculator should need nextn embeddings"
    );

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(args.n_ctx as usize, 1);

    // prefill the target; MTP needs logits at every prompt position
    let prompt_head = &tokens[..tokens.len() - 1];
    for (i, t) in prompt_head.iter().enumerate() {
        batch.add(*t, i as i32, &[0], true)?;
    }
    ctx_tgt
        .decode(&mut batch)
        .context("prefill decode failed")?;
    spec.process(&batch).context("prefill process failed")?;
    spec.begin(0, prompt_head)
        .context("speculative begin failed")?;

    let mut id_last = *tokens.last().unwrap();
    let mut n_past = (tokens.len() - 1) as i32;
    let mut prompt_tgt: Vec<LlamaToken> = prompt_head.to_vec();

    let mut n_drafted: u64 = 0;
    let mut n_accept: u64 = 0;
    let mut n_target_decodes: u64 = 0;
    let mut generated = 0i32;

    if !args.quiet {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }
    let t_start = Instant::now();

    'gen: while generated < args.n_predict {
        let draft = spec
            .draft(0, n_past, id_last, &prompt_tgt)
            .context("draft failed")?;
        n_drafted += draft.len() as u64;

        // roll back ctx_dft's tentative draft decode so process() can re-decode authoritatively
        anyhow::ensure!(
            ctx_dft.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?,
            "draft-context rollback failed (n_rs_seq too small for recurrent state?)"
        );

        // verify batch: [id_last, draft0, draft1, ...]
        batch.clear();
        batch.add(id_last, n_past, &[0], true)?;
        for (i, d) in draft.iter().enumerate() {
            batch.add(*d, n_past + 1 + i as i32, &[0], true)?;
        }
        ctx_tgt.decode(&mut batch).context("verify decode failed")?;
        n_target_decodes += 1;
        spec.process(&batch).context("process failed")?;

        // greedy acceptance: longest draft prefix the target agrees with
        let mut ids: Vec<i32> = Vec::with_capacity(draft.len() + 1);
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
        spec.accept(0, n_acc).context("accept failed")?;

        let mut eos = false;
        for &t in &ids {
            let tok = LlamaToken(t);
            if model.is_eog_token(tok) {
                eos = true;
                break;
            }
            emit(model, &mut decoder, tok, args.quiet);
            prompt_tgt.push(id_last);
            id_last = tok;
            n_past += 1;
            generated += 1;
            if generated >= args.n_predict {
                break;
            }
        }

        // trim rejected drafts from both contexts (bounded by n_rs_seq)
        anyhow::ensure!(
            ctx_tgt.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?,
            "target-context draft trim failed (n_rs_seq too small for recurrent state?)"
        );
        anyhow::ensure!(
            ctx_dft.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?,
            "draft-context draft trim failed (n_rs_seq too small for recurrent state?)"
        );

        if eos {
            break 'gen;
        }
    }

    report("mtp", generated, n_drafted, n_accept, n_target_decodes, t_start);
    Ok(())
}

fn report(
    mode: &str,
    generated: i32,
    n_drafted: u64,
    n_accept: u64,
    n_target_decodes: u64,
    t_start: Instant,
) {
    let elapsed = t_start.elapsed().as_secs_f64();
    let speed = f64::from(generated) / elapsed;
    if mode == "mtp" {
        let pct = if n_drafted > 0 {
            100.0 * n_accept as f64 / n_drafted as f64
        } else {
            0.0
        };
        // Tokens emitted per target forward pass — the speculative win. A plain decoder
        // always emits exactly 1.0 token per target decode; > 1.0 means speculation is
        // amortizing the expensive full-model pass over multiple accepted tokens.
        let toks_per_decode = if n_target_decodes > 0 {
            f64::from(generated) / n_target_decodes as f64
        } else {
            0.0
        };
        println!("\n");
        println!(
            "[mtp] generated={generated} n_drafted={n_drafted} n_accept={n_accept} \
             accept={pct:.1}% target_decodes={n_target_decodes} \
             tok/decode={toks_per_decode:.2} time={elapsed:.2}s speed={speed:.1} tok/s"
        );
    } else {
        println!("\n");
        println!("[baseline] generated={generated} time={elapsed:.2}s speed={speed:.1} tok/s");
    }
}
