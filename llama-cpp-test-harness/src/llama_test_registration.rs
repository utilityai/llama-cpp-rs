use crate::context_params::ContextParams;
use crate::llama_test_fn::LlamaTestFn;
use crate::load_key::LoadKey;

pub struct LlamaTestRegistration {
    pub name: &'static str,
    pub key: LoadKey,
    pub context_params: ContextParams,
    pub void_logs: bool,
    pub func: LlamaTestFn,
}

inventory::collect!(LlamaTestRegistration);
