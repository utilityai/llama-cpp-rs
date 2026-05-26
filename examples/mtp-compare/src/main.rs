//! Compare plain autoregressive decoding against Qwen3.5 MTP decoding.
//!
//! Example:
//!
//! ```console
//! cargo run -p mtp-compare --release --features cuda -- \
//!   --prompt "Write a short poem about Rust" \
//!   --n-predict 64 \
//!   --draft-n 4 \
//!   hf-model unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-Q4_K_M.gguf
//! ```
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::io::Write;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::timing::LlamaTimings;
use llama_cpp_2::token::LlamaToken;

#[derive(Parser, Debug, Clone)]
struct Args {
    #[command(subcommand)]
    model: ModelSource,
    #[arg(short = 'p', long, default_value = "Write a short poem about Rust.")]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    n_predict: usize,
    #[arg(long, default_value_t = 4)]
    draft_n: usize,
    #[arg(short = 'c', long)]
    ctx_size: Option<NonZeroU32>,
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[arg(long)]
    disable_gpu: bool,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum ModelSource {
    Local {
        path: PathBuf,
    },
    #[clap(name = "hf-model")]
    HuggingFace {
        repo: String,
        model: String,
    },
}

impl ModelSource {
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Self::Local { path } => Ok(path),
            Self::HuggingFace { repo, model } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create Hugging Face API")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

#[derive(Debug)]
struct RunStats {
    mode: &'static str,
    output: String,
    wall_ms: f64,
    generated_tokens: usize,
    prompt_tokens: usize,
    timings: LlamaTimings,
    drafted_tokens: usize,
    accepted_tokens: usize,
}

impl RunStats {
    fn wall_tps(&self) -> f64 {
        if self.wall_ms <= 0.0 {
            0.0
        } else {
            1e3 * self.generated_tokens as f64 / self.wall_ms
        }
    }

    fn eval_tps(&self) -> f64 {
        if self.timings.t_eval_ms() <= 0.0 {
            0.0
        } else {
            1e3 * self.timings.n_eval() as f64 / self.timings.t_eval_ms()
        }
    }
}

fn main() -> Result<()> {
    let Args {
        model,
        prompt,
        n_predict,
        draft_n,
        ctx_size,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
    } = Args::parse();

    let model_path = model.get_or_load()?;
    ensure_model_has_mtp(&model_path)?;

    let backend = LlamaBackend::init()?;
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("unable to load model from {}", model_path.display()))?;

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| "failed to tokenize prompt")?;
    if prompt_tokens.is_empty() {
        bail!("prompt tokenized to an empty token list");
    }

    let plain = run_plain(&backend, &model, &prompt_tokens, n_predict, ctx_size)?;
    let mtp = run_mtp(
        &backend,
        &model,
        &prompt_tokens,
        n_predict,
        draft_n,
        ctx_size,
    )?;

    print_stats(&plain);
    println!();
    print_stats(&mtp);
    println!();
    println!(
        "Speedup: wall {:.2}x, eval {:.2}x",
        safe_ratio(mtp.wall_tps(), plain.wall_tps()),
        safe_ratio(mtp.eval_tps(), plain.eval_tps())
    );

    Ok(())
}

fn print_stats(stats: &RunStats) {
    println!("== {} ==", stats.mode);
    println!("Prompt tokens: {}", stats.prompt_tokens);
    println!("Generated tokens: {}", stats.generated_tokens);
    println!("Wall time: {:.2} ms", stats.wall_ms);
    println!("Wall throughput: {:.2} tok/s", stats.wall_tps());
    println!("Eval throughput: {:.2} tok/s", stats.eval_tps());
    println!(
        "Prompt eval: {:.2} ms over {} tokens",
        stats.timings.t_p_eval_ms(),
        stats.timings.n_p_eval()
    );
    println!(
        "Eval: {:.2} ms over {} tokens",
        stats.timings.t_eval_ms(),
        stats.timings.n_eval()
    );
    if stats.drafted_tokens > 0 {
        println!(
            "Drafted tokens: {}, accepted tokens: {}, acceptance rate: {:.2}%",
            stats.drafted_tokens,
            stats.accepted_tokens,
            100.0 * stats.accepted_tokens as f64 / stats.drafted_tokens as f64
        );
    }
    println!("Output:\n{}", stats.output);
}

fn run_plain(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt_tokens: &[LlamaToken],
    n_predict: usize,
    ctx_size: Option<NonZeroU32>,
) -> Result<RunStats> {
    let n_batch = prompt_tokens.len() + n_predict + 1;
    let n_batch_u32 = u32::try_from(n_batch).with_context(|| "batch size does not fit into u32")?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(ctx_size)
        .with_n_batch(n_batch_u32)
        .with_n_ubatch(n_batch_u32);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .with_context(|| "failed to create plain context")?;
    ctx.reset_timings();

    let start = Instant::now();
    prefill_target(&mut ctx, prompt_tokens)?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let eos = model.token_eos();
    let mut generated_tokens = 0usize;
    let mut next_pos = prompt_tokens.len();

    while generated_tokens < n_predict {
        let token = greedy_token(ctx.get_logits());
        if token == eos {
            break;
        }

        append_token(model, &mut decoder, &mut output, token)?;
        generated_tokens += 1;

        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(
                token,
                i32::try_from(next_pos).expect("position fits into i32"),
                &[0],
                true,
            )
            .map_err(anyhow::Error::from)?;
        ctx.decode(&mut batch)
            .with_context(|| "plain decode step failed")?;
        next_pos += 1;
    }

    Ok(RunStats {
        mode: "Plain",
        output,
        wall_ms: start.elapsed().as_secs_f64() * 1e3,
        generated_tokens,
        prompt_tokens: prompt_tokens.len(),
        timings: ctx.timings(),
        drafted_tokens: 0,
        accepted_tokens: 0,
    })
}

fn run_mtp(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt_tokens: &[LlamaToken],
    n_predict: usize,
    draft_n: usize,
    ctx_size: Option<NonZeroU32>,
) -> Result<RunStats> {
    let n_batch = prompt_tokens.len() + n_predict + draft_n + 1;
    let n_batch_u32 = u32::try_from(n_batch).with_context(|| "batch size does not fit into u32")?;
    let n_rs_seq_u32 = u32::try_from(draft_n).with_context(|| "draft_n does not fit into u32")?;

    let base_ctx_params = LlamaContextParams::default()
        .with_n_ctx(ctx_size)
        .with_n_rs_seq(n_rs_seq_u32)
        .with_n_batch(n_batch_u32)
        .with_n_ubatch(n_batch_u32);

    let mut target = model
        .new_context(backend, base_ctx_params.clone())
        .with_context(|| "failed to create target context")?;
    let mut draft = model
        .new_context(
            backend,
            base_ctx_params
                .with_ctx_type(LlamaContextType::Mtp)
                .with_n_rs_seq(n_rs_seq_u32),
        )
        .with_context(|| "failed to create MTP draft context")?;

    target
        .set_embeddings_pre_norm(true)
        .with_context(|| "failed to enable target pre-norm embeddings")?;
    draft
        .set_embeddings_pre_norm(true)
        .with_context(|| "failed to enable MTP draft pre-norm embeddings")?;

    target.reset_timings();
    draft.reset_timings();

    let start = Instant::now();
    let eos = model.token_eos();
    let n_embd =
        usize::try_from(model.n_embd()).with_context(|| "n_embd does not fit into usize")?;

    prefill_target(&mut target, prompt_tokens)?;
    prefill_mtp_context(&mut draft, &target, prompt_tokens, n_embd)?;

    let mut tokens = prompt_tokens.to_vec();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut generated_tokens = 0usize;
    let mut drafted_tokens = 0usize;
    let mut accepted_tokens = 0usize;

    while generated_tokens < n_predict {
        let prefix_hidden = target.embeddings_pre_norm()?.to_vec();
        let drafted = draft_tokens(&mut draft, n_embd, tokens.len(), draft_n, eos)?;
        if drafted.is_empty() {
            break;
        }
        drafted_tokens += drafted.len();

        let accepted = accept_tokens(
            &mut target,
            &mut draft,
            &drafted,
            tokens.len(),
            n_embd,
            &prefix_hidden,
        )?;
        accepted_tokens += accepted.len().min(drafted.len());

        for token in accepted {
            if token == eos || generated_tokens >= n_predict {
                return Ok(RunStats {
                    mode: "MTP",
                    output,
                    wall_ms: start.elapsed().as_secs_f64() * 1e3,
                    generated_tokens,
                    prompt_tokens: prompt_tokens.len(),
                    timings: target.timings(),
                    drafted_tokens,
                    accepted_tokens,
                });
            }

            append_token(model, &mut decoder, &mut output, token)?;
            tokens.push(token);
            generated_tokens += 1;

            if generated_tokens >= n_predict {
                break;
            }
        }
    }

    Ok(RunStats {
        mode: "MTP",
        output,
        wall_ms: start.elapsed().as_secs_f64() * 1e3,
        generated_tokens,
        prompt_tokens: prompt_tokens.len(),
        timings: target.timings(),
        drafted_tokens,
        accepted_tokens,
    })
}

fn ensure_model_has_mtp(path: &Path) -> Result<()> {
    let gguf = GgufContext::from_file(path)
        .with_context(|| format!("unable to inspect GGUF metadata at {}", path.display()))?;

    let arch_key = gguf.find_key("general.architecture");
    if arch_key < 0 {
        bail!("GGUF is missing general.architecture");
    }

    let arch = gguf
        .val_str(arch_key)
        .with_context(|| "general.architecture is not a valid string")?;
    let nextn_key = format!("{arch}.nextn_predict_layers");
    let nextn_idx = gguf.find_key(&nextn_key);

    if nextn_idx < 0 {
        bail!("model does not advertise bundled MTP layers; expected GGUF key `{nextn_key}`");
    }

    if gguf.val_u32(nextn_idx) == 0 {
        bail!("GGUF key `{nextn_key}` is present but disabled");
    }

    Ok(())
}

fn prefill_target(ctx: &mut LlamaContext, tokens: &[LlamaToken]) -> Result<()> {
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch
        .add_sequence(tokens, 0, true)
        .map_err(anyhow::Error::from)?;
    ctx.decode(&mut batch)
        .with_context(|| "target prefill failed")?;
    Ok(())
}

fn prefill_mtp_context(
    draft: &mut LlamaContext,
    target: &LlamaContext,
    tokens: &[LlamaToken],
    n_embd: usize,
) -> Result<()> {
    let mut batch = LlamaBatch::new_with_embeddings(tokens.len(), n_embd, 1);
    let zero = vec![0.0_f32; n_embd];

    for (i, token) in tokens.iter().enumerate() {
        let embd = if i == 0 {
            zero.as_slice()
        } else {
            target.embeddings_pre_norm_ith(i32::try_from(i - 1).expect("index fits into i32"))?
        };
        let logits = i + 1 == tokens.len();
        batch
            .add_with_embedding(
                *token,
                embd,
                i32::try_from(i).expect("position fits into i32"),
                &[0],
                logits,
            )
            .map_err(anyhow::Error::from)?;
    }

    draft
        .decode(&mut batch)
        .with_context(|| "MTP draft prefill failed")?;
    Ok(())
}

fn draft_tokens(
    draft: &mut LlamaContext,
    n_embd: usize,
    prompt_len: usize,
    draft_n: usize,
    eos: LlamaToken,
) -> Result<Vec<LlamaToken>> {
    let mut drafted = Vec::with_capacity(draft_n);
    let mut h_prev = draft.embeddings_pre_norm()?.to_vec();

    for step in 0..draft_n {
        let token = greedy_token(draft.get_logits());
        drafted.push(token);
        if token == eos {
            break;
        }

        let mut batch = LlamaBatch::new_with_embeddings(1, n_embd, 1);
        batch
            .add_with_embedding(
                token,
                &h_prev,
                i32::try_from(prompt_len + step).expect("position fits into i32"),
                &[0],
                true,
            )
            .map_err(anyhow::Error::from)?;

        draft
            .decode(&mut batch)
            .with_context(|| "MTP draft step failed")?;
        h_prev = draft.embeddings_pre_norm()?.to_vec();
    }

    Ok(drafted)
}

fn accept_tokens(
    target: &mut LlamaContext,
    draft: &mut LlamaContext,
    drafted: &[LlamaToken],
    prompt_len: usize,
    n_embd: usize,
    prefix_hidden: &[f32],
) -> Result<Vec<LlamaToken>> {
    let first = greedy_token(target.get_logits());
    if drafted.first().copied() != Some(first) {
        rollback_and_advance(target, draft, prompt_len, 0, first, n_embd, prefix_hidden)?;
        return Ok(vec![first]);
    }

    let mut batch = LlamaBatch::new(drafted.len(), 1);
    for (i, token) in drafted.iter().enumerate() {
        batch
            .add(
                *token,
                i32::try_from(prompt_len + i).expect("position fits into i32"),
                &[0],
                true,
            )
            .map_err(anyhow::Error::from)?;
    }

    target
        .decode(&mut batch)
        .with_context(|| "target verification decode failed")?;

    let mut matched = 0usize;
    let mut extra = first;

    for i in 0..drafted.len() {
        let sampled = greedy_token(target.get_logits_ith(i32::try_from(i).expect("index fits")));
        if i + 1 == drafted.len() || sampled != drafted[i + 1] {
            matched = i + 1;
            extra = sampled;
            break;
        }
    }

    let mut accepted = drafted[..matched].to_vec();
    accepted.push(extra);

    let extra_hidden = if matched == 0 {
        prefix_hidden.to_vec()
    } else {
        target
            .embeddings_pre_norm_ith(i32::try_from(matched - 1).expect("index fits into i32"))?
            .to_vec()
    };

    rollback_and_advance(
        target,
        draft,
        prompt_len,
        matched,
        extra,
        n_embd,
        &extra_hidden,
    )?;

    Ok(accepted)
}

fn rollback_and_advance(
    target: &mut LlamaContext,
    draft: &mut LlamaContext,
    prompt_len: usize,
    matched: usize,
    extra: LlamaToken,
    n_embd: usize,
    extra_hidden: &[f32],
) -> Result<()> {
    let rollback_pos = u32::try_from(prompt_len + matched)
        .with_context(|| "rollback position does not fit into u32")?;

    let target_rolled_back = target
        .clear_kv_cache_seq(Some(0), Some(rollback_pos), None)
        .with_context(|| "failed to roll back target KV cache")?;
    if !target_rolled_back {
        bail!("target KV rollback failed at position {rollback_pos}");
    }

    let draft_rolled_back = draft
        .clear_kv_cache_seq(Some(0), Some(rollback_pos), None)
        .with_context(|| "failed to roll back MTP draft KV cache")?;
    if !draft_rolled_back {
        bail!("MTP draft KV rollback failed at position {rollback_pos}");
    }

    let extra_pos = i32::try_from(prompt_len + matched)
        .with_context(|| "extra token position does not fit into i32")?;

    let mut target_batch = LlamaBatch::new(1, 1);
    target_batch
        .add(extra, extra_pos, &[0], true)
        .map_err(anyhow::Error::from)?;
    target
        .decode(&mut target_batch)
        .with_context(|| "failed to advance target with accepted token")?;

    let mut draft_batch = LlamaBatch::new_with_embeddings(1, n_embd, 1);
    draft_batch
        .add_with_embedding(extra, extra_hidden, extra_pos, &[0], true)
        .map_err(anyhow::Error::from)?;
    draft
        .decode(&mut draft_batch)
        .with_context(|| "failed to advance MTP draft with accepted token")?;

    Ok(())
}

fn greedy_token(logits: &[f32]) -> LlamaToken {
    let (idx, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("logits must not be empty");
    LlamaToken::new(i32::try_from(idx).expect("token index fits into i32"))
}

fn append_token(
    model: &LlamaModel,
    decoder: &mut encoding_rs::Decoder,
    output: &mut String,
    token: LlamaToken,
) -> Result<()> {
    let piece = model
        .token_to_piece(token, decoder, true, None)
        .with_context(|| format!("failed to decode token {token}"))?;
    output.push_str(&piece);
    std::io::stdout().flush()?;
    Ok(())
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
