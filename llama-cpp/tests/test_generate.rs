use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::LlamaModel;
use llama_cpp::token::data_array::LlamaTokenDataArray;
use std::error::Error;
use std::io;
use std::io::Write;
use std::num::NonZeroU32;

#[test]
#[ignore] // slow
fn check_special_token_to_str() -> Result<(), Box<dyn Error>> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;
    let file = api
        .model("TheBloke/Llama-2-7b-Chat-GGUF".to_string())
        .get("llama-2-7b-chat.Q4_K_M.gguf")?;
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_vocab_only(true);
    let model = LlamaModel::load_from_file(&backend, &file, &model_params)?;
    let ctx = model.new_context(&backend, &LlamaContextParams::default())?;

    let model = ctx.model;
    let n_vocab = model.n_vocab();

    println!("n_vocab = {n_vocab}");
    Ok(())
}

#[test]
#[ignore] // slow
fn check_generate_tokens() -> Result<(), Box<dyn Error>> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;
    let file = api
        .model("TheBloke/Llama-2-7b-Chat-GGUF".to_string())
        .get("llama-2-7b-chat.Q2_K.gguf")?;
    let n_len = 64;
    let prompt =
        "[INST] <<SYS>>You are a helpful, respectful and honest assistant.<</SYS>>Hello![/INST]";
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &file, &model_params)?;
    let tokens_list = model.str_to_token(prompt, true)?;
    let ctx_params = LlamaContextParams {
        seed: 1234,
        n_ctx: Some(NonZeroU32::new(u32::try_from(n_len).unwrap()).unwrap()),
        n_batch: u32::try_from(n_len).unwrap(),
        n_threads: u32::try_from(std::thread::available_parallelism().unwrap().get()).unwrap(),
        n_threads_batch: 1,
        ..LlamaContextParams::default()
    };
    let mut ctx = model.new_context(&backend, &ctx_params)?;
    let n_ctx = ctx.n_ctx();
    println!(
        "n_len = {}, n_ctx = {}, n_batch = {}",
        n_ctx, n_ctx, ctx_params.n_batch
    );
    let mut stdoutlock = io::stdout().lock();
    for id in &tokens_list {
        write!(stdoutlock, "{}", model.token_to_str(*id)?)?;
    }
    stdoutlock.flush()?;
    let mut batch = LlamaBatch::new(tokens_list.len(), 0, 1);
    batch.add_prompt_seq(&tokens_list, &[0]);
    ctx.decode(&mut batch)?;
    let mut n_cur = batch.n_tokens();
    while n_cur < n_len {
        batch.clear();
        let candidates_p = LlamaTokenDataArray::from_iter(ctx.candidates_ith(1), false);
        let new_token_id = ctx.sample_token_greedy(candidates_p);
        if new_token_id == model.token_eos() {
            break;
        }
        write!(stdoutlock, "{}", model.token_to_str(new_token_id)?)?;
        stdoutlock.flush()?;
        batch.add(new_token_id, n_cur, &[0i32], true);
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }
    println!("\n{}", ctx.timings());
    println!("{:?}", model.token_eos());

    Ok(())
}
