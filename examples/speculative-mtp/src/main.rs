//! MTP (Multi-Token Prediction / NextN) speculative decoding example.
//!
//! Runs a single-sequence speculative loop using a single GGUF whose embedded NextN head drafts
//! tokens from the target model's own hidden state. Acceptance is greedy (argmax) and done in
//! Rust; the MTP head forward runs inside llama.cpp via the `MtpSpeculator` shim.
//!
//! Doubles as the smoke test: it prints the generated text and the draft acceptance rate.
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
#[command(about = "MTP speculative decoding smoke test")]
struct Args {
    /// Path to the MTP GGUF (with embedded NextN head).
    #[arg(long)]
    model: PathBuf,
    /// Prompt.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// Max tokens to generate.
    #[arg(long, default_value_t = 64)]
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

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .context("failed to load model")?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(args.n_ctx));

    // Target context, then the MTP draft context on the same model (shares memory with target).
    let mut ctx_tgt = model
        .new_context(&backend, ctx_params.clone())
        .context("failed to create target context")?;
    let mut ctx_dft = model
        .new_mtp_context(&backend, &ctx_tgt, ctx_params)
        .context("failed to create MTP draft context")?;

    let mut spec = MtpSpeculator::new(&ctx_tgt, &ctx_dft, args.n_draft, 0, 0.0, true)
        .context("failed to init MTP speculator")?;
    assert!(spec.need_embd_nextn(), "MTP speculator should need nextn embeddings");

    let tokens = model
        .str_to_token(&args.prompt, AddBos::Always)
        .context("failed to tokenize prompt")?;
    anyhow::ensure!(tokens.len() >= 2, "prompt must be at least 2 tokens");

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(args.n_ctx as usize, 1);

    // Prefill the target on all-but-last prompt token. MTP requires logits at every prompt
    // position so the speculator can mirror the target hidden state into the draft context.
    let prompt_head = &tokens[..tokens.len() - 1];
    for (i, t) in prompt_head.iter().enumerate() {
        batch.add(*t, i as i32, &[0], true)?;
    }
    ctx_tgt.decode(&mut batch).context("prefill decode failed")?;
    spec.process(&batch).context("prefill process failed")?;
    spec.begin(0, prompt_head).context("speculative begin failed")?;

    let mut id_last = *tokens.last().unwrap();
    let mut n_past = (tokens.len() - 1) as i32;
    let mut prompt_tgt: Vec<LlamaToken> = prompt_head.to_vec();

    let mut n_drafted: u64 = 0;
    let mut n_accept: u64 = 0;
    let mut generated = 0i32;

    print!("{}", args.prompt);
    std::io::stdout().flush().ok();

    let t_start = Instant::now();

    'gen: while generated < args.n_predict {
        // 1. draft from the current last token
        let draft = spec
            .draft(0, n_past, id_last, &prompt_tgt)
            .context("draft failed")?;
        n_drafted += draft.len() as u64;

        // 2. build + decode the verify batch: [id_last, draft0, draft1, ...]
        batch.clear();
        batch.add(id_last, n_past, &[0], true)?;
        for (i, d) in draft.iter().enumerate() {
            batch.add(*d, n_past + 1 + i as i32, &[0], true)?;
        }
        ctx_tgt.decode(&mut batch).context("verify decode failed")?;

        // 3. let the speculator capture the target hidden states for the next draft.
        // For non-shared-memory archs (e.g. qwen35) draft() tentatively decoded the draft
        // tokens into ctx_dft at [n_past+1, n_past+N]; process() re-decodes the verify batch
        // into ctx_dft authoritatively, so clear those tentative positions first.
        ctx_dft.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?;
        spec.process(&batch).context("process failed")?;

        // 4. greedy acceptance in Rust: accept the longest draft prefix the target agrees with
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

        // 5. tell the speculator how many drafts were accepted (excludes the bonus token)
        spec.accept(0, n_acc).context("accept failed")?;

        // 6. commit accepted tokens
        let mut eos = false;
        for &t in &ids {
            let tok = LlamaToken(t);
            if model.is_eog_token(tok) {
                eos = true;
                break;
            }
            let piece = model.token_to_piece(tok, &mut decoder, true, None)?;
            print!("{piece}");
            std::io::stdout().flush().ok();
            prompt_tgt.push(id_last);
            id_last = tok;
            n_past += 1;
            generated += 1;
            if generated >= args.n_predict {
                break;
            }
        }

        // 7. trim the rejected draft tokens from both contexts' KV
        ctx_tgt.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?;
        ctx_dft.clear_kv_cache_seq(Some(0), Some(n_past as u32), None)?;

        if eos {
            break 'gen;
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let pct = if n_drafted > 0 {
        100.0 * n_accept as f64 / n_drafted as f64
    } else {
        0.0
    };
    println!("\n");
    println!(
        "[mtp] generated={generated} n_drafted={n_drafted} n_accept={n_accept} accept={pct:.1}% \
         time={elapsed:.2}s speed={:.1} tok/s",
        f64::from(generated) / elapsed
    );
    Ok(())
}
