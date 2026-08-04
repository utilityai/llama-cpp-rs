//! Checks that the grammar samplers work when the `common` feature is off.
//!
//! Set `LLAMA_TEST_VOCAB_GGUF` to a GGUF file to run these.

use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::GrammarError;

const GRAMMAR: &str = r#"root ::= "{" "\"a\"" ":" [0-9] "}""#;

/// The vocabulary given by `LLAMA_TEST_VOCAB_GGUF`, or `None` to skip.
///
/// The backend can be started only one time in a process, and the tests share
/// this process, so both the backend and the model are made one time here.
/// `vocab_only` keeps the load cheap: a grammar needs the token table and no
/// tensor data.
fn model() -> Option<&'static LlamaModel> {
    static MODEL: OnceLock<Option<(LlamaBackend, LlamaModel)>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = std::env::var("LLAMA_TEST_VOCAB_GGUF").ok()?;
            let backend = LlamaBackend::init().unwrap();
            let params = LlamaModelParams::default().with_vocab_only(true);
            let model = LlamaModel::load_from_file(&backend, path, &params).unwrap();
            Some((backend, model))
        })
        .as_ref()
        .map(|(_backend, model)| model)
}

#[test]
fn grammar_is_accepted() {
    let Some(model) = model() else {
        return;
    };
    assert!(LlamaSampler::grammar(model, GRAMMAR, "root").is_ok());
}

#[test]
fn lazy_patterns_grammar_is_accepted() {
    let Some(model) = model() else {
        return;
    };
    let patterns = vec![r"\{".to_string()];
    assert!(LlamaSampler::grammar_lazy_patterns(model, GRAMMAR, "root", &patterns, &[]).is_ok());
}

#[test]
fn unparseable_grammar_gives_null_grammar() {
    let Some(model) = model() else {
        return;
    };
    assert_eq!(
        LlamaSampler::grammar(model, "root ::= <<<", "root").err(),
        Some(GrammarError::NullGrammar)
    );
}

#[test]
fn grammar_root_is_checked() {
    let Some(model) = model() else {
        return;
    };
    assert_eq!(
        LlamaSampler::grammar(model, GRAMMAR, "missing").err(),
        Some(GrammarError::RootNotFound)
    );
}
