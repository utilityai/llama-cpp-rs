#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::grammar::LlamaGrammar;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use pprof::criterion::{Output, PProfProfiler};
use std::str::FromStr;

fn sample(
    llama_grammar: &LlamaGrammar,
    candidates: &mut LlamaTokenDataArray,
    llama_context: &mut LlamaContext,
) {
    llama_context.sample_grammar(candidates, llama_grammar);
}

fn criterion_benchmark(c: &mut Criterion) {
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
    let model = LlamaModel::load_from_file(&backend, file, &model_params).unwrap();
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .unwrap();
    let grammar = LlamaGrammar::from_str(include_str!("../src/grammar/json.gbnf")).unwrap();

    c.bench_function("sample grammar", |b| {
        b.iter(|| {
            let mut candidates = LlamaTokenDataArray::from_iter(ctx.candidates_ith(0), false);
            sample(&grammar, &mut candidates, &mut ctx);
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
);
criterion_main!(benches);
