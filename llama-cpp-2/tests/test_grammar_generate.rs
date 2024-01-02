use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::grammar::LlamaGrammar;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::LlamaModel;
use llama_cpp::token::data_array::LlamaTokenDataArray;

use llama_cpp_sys_2::llama_pos;
use std::str::FromStr;

#[test]
fn test_generate_cat_via_grammar() {
    let grammar = r#"root ::= "cat""#;
    let mut grammar = LlamaGrammar::from_str(grammar).unwrap();

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .unwrap();
    let file = api
        .model("TheBloke/Llama-2-7b-Chat-GGUF".to_string())
        .get("llama-2-7b-chat.Q4_K_M.gguf")
        .unwrap();
    let backend = LlamaBackend::init().unwrap();
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &file, &model_params).unwrap();

    let mut ctx = model
        .new_context(&backend, &LlamaContextParams::default())
        .unwrap();

    let n_ctx_usize = usize::try_from(ctx.n_ctx()).expect("cannot fit n_ctx into a usize");
    let mut batch = LlamaBatch::new(n_ctx_usize, 0, 1);
    batch.add(model.token_bos(), 0, &[0], true);

    let mut tokens = vec![model.token_bos()];
    loop {
        ctx.decode(&mut batch).unwrap();
        batch.clear();
        let mut candidates = LlamaTokenDataArray::from_iter(ctx.candidates_ith(0), false);

        ctx.sample_grammar(&mut candidates, &grammar);
        let token = ctx.sample_token_greedy(candidates);
        if token == model.token_eos() {
            break;
        }
        ctx.grammar_accept_token(&mut grammar, token);
        tokens.push(token);
        batch.add(
            token,
            llama_pos::try_from(tokens.len()).unwrap(),
            &[0],
            true,
        );
    }
    let str = model.tokens_to_str(&tokens).unwrap();

    assert_eq!("cat", str);
}
