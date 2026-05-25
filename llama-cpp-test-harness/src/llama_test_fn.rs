use crate::llama_fixture::LlamaFixture;

pub type LlamaTestFn =
    for<'reference, 'fixture> fn(&'reference LlamaFixture<'fixture>) -> anyhow::Result<()>;
