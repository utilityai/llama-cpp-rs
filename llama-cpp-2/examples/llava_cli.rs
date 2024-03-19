//! llava-cli demo

use std::ffi::c_int;
use std::num::NonZeroU32;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CString, pin::pin};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::llava::{
    llava_sample, LlamaSamplingContext, LlamaSamplingParams, LlavaImageEmbed,
};
use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{
    context::LlamaContext,
    llama_backend::LlamaBackend,
    llava::ClipCtx,
    model::{
        params::{kv_overrides::ParamOverrideValue, LlamaModelParams},
        LlamaModel,
    },
};

mod common;

#[derive(Parser, Debug, Clone)]
#[command(
    version,
    about = "llava cli demo",
    long_about = "llava cli demo for Rust"
)]
struct Args {
    /// llama model
    #[arg(short, long)]
    model: String,

    /// clip model
    #[arg(long)]
    mmproj: String,

    /// temperature. Note: a lower temperature value like 0.1 is recommended for better quality.
    #[arg(short, long, default_value_t = 0.1)]
    temperature: f32,

    /// path to image
    #[arg(short, long)]
    image: String,

    /// path to image
    #[arg(short, long, default_value = "describe the image in detail.")]
    prompt: String,

    /// override some parameters of the model
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    // /// Disable offloading layers to the gpu
    // #[cfg(feature = "cublas")]
    // #[clap(long)]
    // disable_gpu: bool,
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

struct LlavaContext {
    ctx_clip: ClipCtx,
    ctx_llama: LlamaContext,
    model: Arc<LlamaModel>,
}

fn llava_init<'a>(args: &'a Args) -> Result<LlavaContext> {
    // locad clip model
    let ctx_clip = ClipCtx::load_from_file(&args.mmproj, 1)?;

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(feature = "cublas")]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default()
    };

    let mut model_params = pin!(model_params);
    for (k, v) in &args.key_value_overrides {
        let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    let llama_model_path = &args.model;
    let llama_model = LlamaModel::load_from_file(&backend, llama_model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_seed(1234);

    let ctx_llama = llama_model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    Ok(LlavaContext {
        ctx_clip,
        ctx_llama: ctx_llama,
        model: llama_model,
    })
}

fn load_image(ctx_llava: &mut LlavaContext, args: &Args) -> Result<LlavaImageEmbed> {
    let embed = LlavaImageEmbed::make_with_file(&mut ctx_llava.ctx_clip, 4, &args.image)?;

    Ok(embed)
}

fn process_prompt(
    ctx_llava: &mut LlavaContext,
    image_embed: &LlavaImageEmbed,
    args: &Args,
    prompt: &str,
) -> Result<()> {
    let system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
    let user_prompt = format!("{}\nASSISTANT:", prompt);

    let mut n_past: c_int = 0;
    let n_batch = 2048; // logical batch size for prompt processing (must be >=32 to use BLAS)

    // eval system prompt
    eval_string(
        &mut ctx_llava.ctx_llama,
        system_prompt,
        n_batch,
        &mut n_past,
        AddBos::Always,
    )?;
    // eval image
    image_embed.eval(&mut ctx_llava.ctx_llama, n_batch, &mut n_past);
    // eval user prompt
    eval_string(
        &mut ctx_llava.ctx_llama,
        &user_prompt,
        n_batch,
        &mut n_past,
        AddBos::Never,
    )?;

    // generate the response
    eprintln!();
    let params_sampling = LlamaSamplingParams::default()?;
    let mut ctx_sampling = LlamaSamplingContext::init(&params_sampling)?;
    let mut response = "".to_owned();
    const max_tgt_len: c_int = 256;

    for i in 0..max_tgt_len {
        let tmp = llava_sample(&mut ctx_sampling, &mut ctx_llava.ctx_llama, &mut n_past);
        response.push_str(&tmp);
        if tmp == "</s>" {
            break;
        }
        if tmp.contains("###") {
            // Yi-VL behavior
            break;
        }
        print!("{}", tmp);
        if response.contains("<|im_end|>") {
            // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
            break;
        }
        if response.contains("<|im_start|>") {
            // // Yi-34B llava-1.6
            break;
        }
        if response.contains("USER:") {
            // mistral llava-1.6
            break;
        }
        println!();
    }

    Ok(())
}

fn eval_string(
    ctx_llama: &mut LlamaContext,
    str: &str,
    n_batch: i32,
    n_past: &mut i32,
    add_bos: AddBos,
) -> Result<bool> {
    let embd_inp = ctx_llama.model.str_to_token(str, add_bos)?;

    eval_token(ctx_llama, &embd_inp, n_batch, n_past)
}

fn eval_token(
    ctx_llama: &mut LlamaContext,
    tokens: &[LlamaToken],
    n_batch: c_int,
    n_past: &mut i32,
) -> Result<bool> {
    let n = tokens.len() as c_int;
    let mut i: c_int = 0;
    while i < n {
        let mut n_eval = (tokens.len() as c_int) - i;
        if n_eval > n_batch {
            n_eval = n_batch;
        }

        let tokens_batch = &tokens[i as usize..(i + n_eval) as usize];
        let mut batch = LlamaBatch::get_one(&tokens_batch, *n_past, 0);
        ctx_llama.decode(&mut batch).with_context(|| {
            format!(
                "failed to eval. token {}/{} (batch size {}, n_past {})",
                i, n, n_batch, n_past,
            )
        })?;

        *n_past += n_eval;
        i += n_batch;
    }

    Ok(true)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("args: {:#?}", args);

    let mut ctx_llava = llava_init(&args)?;

    let image_embed = load_image(&mut ctx_llava, &args)?;

    process_prompt(&mut ctx_llava, &image_embed, &args, &args.prompt)?;

    Ok(())
}
