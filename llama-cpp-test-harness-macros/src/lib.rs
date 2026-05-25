//! Procedural macros for `llama-cpp-test-harness`.
//!
//! Provides the `#[llama_test(...)]` attribute that declaratively binds a test function to a
//! specific GGUF model and inference parameter set. The macro emits the original function plus
//! an `inventory::submit!` block that registers the test with the harness runtime.

mod expand;
mod parsed_args;
mod parsed_context_params;
mod parsed_model_load_params;
mod parsed_source;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

use crate::expand::expand;

fn dispatch(attribute: TokenStream2, item: TokenStream2) -> TokenStream2 {
    match expand(attribute, item) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

/// Registers a function as a llama-cpp test with explicit model + inference parameters.
///
/// See the `llama-cpp-test-harness` crate for the full attribute schema and usage.
#[proc_macro_attribute]
pub fn llama_test(attribute: TokenStream, item: TokenStream) -> TokenStream {
    dispatch(attribute.into(), item.into()).into()
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::dispatch;

    #[test]
    fn dispatch_on_invalid_attribute_emits_compile_error_tokens() {
        let attribute = quote! { totally_wrong = "x" };
        let item = quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let emitted = dispatch(attribute, item).to_string();

        assert!(
            emitted.contains("compile_error"),
            "expected compile_error! tokens in emitted output: {emitted}",
        );
    }

    #[test]
    fn dispatch_on_valid_input_emits_inventory_submission() {
        let attribute = quote! {
            model_source = HuggingFace("r", "f"),
            n_gpu_layers = 0,
            use_mmap = true,
            use_mlock = false,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1
        };
        let item = quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let emitted = dispatch(attribute, item).to_string();

        assert!(
            emitted.contains("inventory"),
            "expected inventory::submit in emitted output: {emitted}",
        );
        assert!(
            !emitted.contains("compile_error"),
            "valid input should not emit compile_error: {emitted}",
        );
    }
}
