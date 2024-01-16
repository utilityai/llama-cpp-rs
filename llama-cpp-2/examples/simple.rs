//! This is an translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use anyhow::{bail, Context, Result};
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::model::AddBos;


#[derive(clap::Parser)]
struct Args {
    /// The path to the model
    model_path: PathBuf,
    /// The prompt
    #[clap(default_value = "Hello my name is")]
    prompt: String,
    /// Disable offloading layers to the gpu
    #[cfg(feature = "cublas")]
    #[clap(long)]
    disable_gpu: bool,
}


fn main() -> Result<()> {
    let params = Args::parse();

    // init LLM
    let backend = LlamaBackend::init()?;

    // total length of the sequence including the prompt
    let n_len: i32 = 32;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(feature = "cublas")]
        if !params.disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default()
    };

    let model = LlamaModel::load_from_file(&backend, params.model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams {
        seed: 1234,
        n_ctx: NonZeroU32::new(2048),
        ..LlamaContextParams::default()
    };

    let mut ctx = model.new_context(&backend, &ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // tokenize the prompt

    let tokens_list = model.str_to_token(&params.prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {}", params.prompt))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    eprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_cxt {
        bail!("n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx")
    }

    // print the prompt token-by-token
    eprintln!();

    for token in &tokens_list {
        eprint!("{}", model.token_to_str(*token)?);
    }

    std::io::stderr().flush()?;

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(512, 1);

    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last);
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // main loop

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();

    while n_cur <= n_len {
        // sample the next token
        {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // sample the most likely token
            let new_token_id = ctx.sample_token_greedy(candidates_p);

            // is it an end of stream?
            if new_token_id == model.token_eos() {
                eprintln!();
                break;
            }

            print!("{}", model.token_to_str(new_token_id)?);
            std::io::stdout().flush()?;

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true);
        }

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;

        n_decode += 1;

    }

    eprintln!("\n");

    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    eprintln!("decoded {} tokens in {:.2} s, speed {:.2} t/s\n", n_decode, duration.as_secs_f32(), n_decode as f32 / duration.as_secs_f32());

    println!("{}", ctx.timings());

    Ok(())

}