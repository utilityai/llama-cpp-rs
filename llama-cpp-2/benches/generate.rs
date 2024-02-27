use anyhow::Context;
use criterion::{criterion_group, criterion_main, Criterion};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use pprof::criterion::{Output, PProfProfiler};

fn generate(c: &mut Criterion) {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .unwrap();
    let file = api
        .model("TheBloke/Llama-2-7B-Chat-GGUF".to_string())
        .get("llama-2-7b-chat.Q4_K_M.gguf")
        .unwrap();
    let backend = LlamaBackend::init().unwrap();
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &file, &model_params).unwrap();
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .unwrap();

    c.bench_function("generate 50 tokens", |b| {
        b.iter(|| {
            let tokens_list = model
                .str_to_token("Hello, my name is", AddBos::Always)
                .unwrap();
            let mut n_ctx = tokens_list.len() as i32;
            let mut batch = LlamaBatch::new(512, 1);
            let last_index: i32 = (tokens_list.len() - 1) as i32;
            for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last).unwrap();
            }
            ctx.decode(&mut batch).unwrap();

            for _ in 0..50 {
                let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
                let new_token_id = ctx.sample_token_greedy(candidates_p);
                if new_token_id == model.token_eos() {
                    break;
                }
                batch.clear();
                batch.add(new_token_id, n_ctx, &[0], true).unwrap();
                n_ctx += 1;
                ctx.decode(&mut batch).unwrap();
            }
            ctx.clear_kv_cache_seq(0, None, None)
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = generate
);
criterion_main!(benches);
