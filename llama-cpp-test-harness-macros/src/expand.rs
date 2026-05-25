use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Item;
use syn::ItemFn;
use syn::ReturnType;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::parse2;

use crate::parsed_args::ParsedArgs;
use crate::parsed_source::ParsedSource;

struct StackedItems {
    items: Vec<Item>,
}

impl Parse for StackedItems {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        Ok(Self { items })
    }
}

fn validate_signature(item_fn: &ItemFn) -> syn::Result<()> {
    if item_fn.sig.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.inputs,
            "llama_test functions must take exactly one argument: `&LlamaFixture<'_>`",
        ));
    }
    if matches!(item_fn.sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "llama_test functions must return `anyhow::Result<()>`",
        ));
    }
    if item_fn.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "llama_test functions must be synchronous",
        ));
    }
    if !item_fn.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.generics,
            "llama_test functions must not be generic",
        ));
    }
    Ok(())
}

fn split_fn_and_pass_through(items: Vec<Item>) -> syn::Result<(ItemFn, Vec<Item>)> {
    let mut found_fn: Option<ItemFn> = None;
    let mut pass_through: Vec<Item> = Vec::new();
    for parsed_item in items {
        match parsed_item {
            Item::Fn(item_fn) => {
                if found_fn.is_some() {
                    return Err(syn::Error::new_spanned(
                        &item_fn,
                        "llama_test expects exactly one fn definition",
                    ));
                }
                found_fn = Some(item_fn);
            }
            other => pass_through.push(other),
        }
    }
    let item_fn = found_fn
        .ok_or_else(|| syn::Error::new(Span::call_site(), "llama_test expects an fn definition"))?;
    Ok((item_fn, pass_through))
}

fn build_model_source_literal(source: &ParsedSource) -> TokenStream {
    match source {
        ParsedSource::HuggingFace { repo, file } => quote! {
            ::llama_cpp_test_harness::ModelSource::HuggingFace {
                repo: #repo,
                file: #file,
            }
        },
        ParsedSource::LocalPath(path) => quote! {
            ::llama_cpp_test_harness::ModelSource::LocalPath(#path)
        },
    }
}

fn build_mmproj_source_literal(source: Option<&ParsedSource>) -> TokenStream {
    match source {
        None => quote! { ::core::option::Option::None },
        Some(ParsedSource::HuggingFace { repo, file }) => quote! {
            ::core::option::Option::Some(::llama_cpp_test_harness::MmprojSource::HuggingFace {
                repo: #repo,
                file: #file,
            })
        },
        Some(ParsedSource::LocalPath(path)) => quote! {
            ::core::option::Option::Some(::llama_cpp_test_harness::MmprojSource::LocalPath(#path))
        },
    }
}

fn build_registration(args: &ParsedArgs, fn_name: &Ident) -> TokenStream {
    let trial_name = format!(
        "{fn_name}[{suffix}]",
        suffix = args.model_source.display_suffix()
    );
    let model_source_literal = build_model_source_literal(&args.model_source);
    let mmproj_source_literal = build_mmproj_source_literal(args.mmproj_source.as_ref());
    let gpu_layers = args.model_load_params.n_gpu_layers;
    let use_mmap = args.model_load_params.use_mmap;
    let use_mlock = args.model_load_params.use_mlock;
    let context_size = args.context_params.n_ctx;
    let logical_batch = args.context_params.n_batch;
    let physical_batch = args.context_params.n_ubatch;
    let embeddings_flag = args.context_params.embeddings;
    let sequence_max = args.context_params.n_seq_max;
    let void_logs_flag = args.void_logs;
    let threads_batch = args.context_params.n_threads_batch.map_or_else(
        || quote! { ::core::option::Option::None },
        |value| quote! { ::core::option::Option::Some(#value) },
    );

    quote! {
        ::llama_cpp_test_harness::inventory::submit! {
            ::llama_cpp_test_harness::LlamaTestRegistration {
                name: #trial_name,
                key: ::llama_cpp_test_harness::LoadKey {
                    model_source: #model_source_literal,
                    mmproj_source: #mmproj_source_literal,
                    model_load_params: ::llama_cpp_test_harness::ModelLoadParams {
                        n_gpu_layers: #gpu_layers,
                        use_mmap: #use_mmap,
                        use_mlock: #use_mlock,
                    },
                },
                context_params: ::llama_cpp_test_harness::ContextParams {
                    n_ctx: #context_size,
                    n_batch: #logical_batch,
                    n_ubatch: #physical_batch,
                    n_seq_max: #sequence_max,
                    n_threads_batch: #threads_batch,
                    embeddings: #embeddings_flag,
                },
                void_logs: #void_logs_flag,
                func: #fn_name,
            }
        }
    }
}

pub fn expand(attribute: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: ParsedArgs = parse2(attribute)?;
    let StackedItems { items } = parse2(item)?;
    let (item_fn, pass_through) = split_fn_and_pass_through(items)?;
    validate_signature(&item_fn)?;

    let fn_name = &item_fn.sig.ident;
    let new_submission = build_registration(&args, fn_name);

    Ok(quote! {
        #item_fn
        #(#pass_through)*
        #new_submission
    })
}

#[cfg(test)]
mod tests {
    use proc_macro2::TokenStream;
    use quote::quote;

    use super::expand;

    fn well_formed_attribute() -> TokenStream {
        quote! {
            model_source = HuggingFace("foo", "bar.gguf"),
            n_gpu_layers = 0,
            use_mmap = true,
            use_mlock = false,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1
        }
    }

    fn well_formed_function() -> TokenStream {
        quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        }
    }

    #[test]
    fn well_formed_input_expands_to_function_plus_submission() {
        let expanded = expand(well_formed_attribute(), well_formed_function())
            .expect("well-formed input must expand")
            .to_string();

        assert!(
            expanded.contains("fn my_test"),
            "expansion missing the original fn: {expanded}"
        );
        assert!(
            expanded.contains("LlamaTestRegistration"),
            "expansion missing LlamaTestRegistration: {expanded}",
        );
        assert!(
            expanded.contains("\"my_test[bar.gguf]\""),
            "expansion missing the trial-name literal with file suffix: {expanded}",
        );
        assert!(
            expanded.contains("ModelSource :: HuggingFace"),
            "expansion missing ModelSource::HuggingFace variant: {expanded}",
        );
        assert!(
            expanded.contains("func : my_test"),
            "expansion missing func wire-up: {expanded}",
        );
    }

    #[test]
    fn expansion_with_local_path_model_source_emits_local_variant() {
        let attribute = quote! {
            model_source = LocalPath("/abs/local.gguf"),
            n_gpu_layers = 0,
            use_mmap = true,
            use_mlock = false,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1
        };
        let expanded = expand(attribute, well_formed_function())
            .expect("LocalPath must expand")
            .to_string();

        assert!(
            expanded.contains("ModelSource :: LocalPath"),
            "expansion missing ModelSource::LocalPath variant: {expanded}",
        );
        assert!(
            expanded.contains("\"my_test[local.gguf]\""),
            "trial name must use the path's filename component: {expanded}",
        );
    }

    #[test]
    fn expansion_with_mmproj_source_emits_some_variant() {
        let attribute = quote! {
            model_source = HuggingFace("r", "f"),
            n_gpu_layers = 0,
            use_mmap = true,
            use_mlock = false,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1,
            mmproj_source = HuggingFace("r", "mmproj.gguf")
        };
        let expanded = expand(attribute, well_formed_function())
            .expect("mmproj_source must expand")
            .to_string();

        assert!(
            expanded.contains("MmprojSource :: HuggingFace"),
            "expansion missing MmprojSource::HuggingFace: {expanded}",
        );
        assert!(
            expanded.contains("Some"),
            "expansion missing Option::Some wrap: {expanded}",
        );
    }

    #[test]
    fn expansion_with_local_path_mmproj_emits_local_variant() {
        let attribute = quote! {
            model_source = HuggingFace("r", "f"),
            n_gpu_layers = 0,
            use_mmap = true,
            use_mlock = false,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1,
            mmproj_source = LocalPath("/abs/mmproj.gguf")
        };
        let expanded = expand(attribute, well_formed_function())
            .expect("LocalPath mmproj must expand")
            .to_string();

        assert!(
            expanded.contains("MmprojSource :: LocalPath"),
            "expansion missing MmprojSource::LocalPath: {expanded}",
        );
    }

    #[test]
    fn expansion_without_mmproj_emits_none_variant() {
        let expanded = expand(well_formed_attribute(), well_formed_function())
            .expect("no-mmproj must expand")
            .to_string();

        assert!(
            expanded.contains("None"),
            "expansion missing Option::None for absent mmproj: {expanded}"
        );
    }

    #[test]
    fn malformed_attribute_propagates_parser_error() {
        let attribute = quote! { totally_wrong = "x" };
        let error = expand(attribute, well_formed_function())
            .expect_err("malformed must fail")
            .to_string();

        assert!(error.contains("unknown field"), "got: {error}");
    }

    #[test]
    fn function_with_too_many_arguments_is_rejected() {
        let item = quote! {
            fn my_test(a: &LlamaFixture<'_>, b: i32) -> anyhow::Result<()> { Ok(()) }
        };
        let error = expand(well_formed_attribute(), item)
            .expect_err("two-arg must fail")
            .to_string();

        assert!(error.contains("exactly one argument"), "got: {error}");
    }

    #[test]
    fn function_with_zero_arguments_is_rejected() {
        let item = quote! { fn my_test() -> anyhow::Result<()> { Ok(()) } };
        let error = expand(well_formed_attribute(), item)
            .expect_err("zero-arg must fail")
            .to_string();

        assert!(error.contains("exactly one argument"), "got: {error}");
    }

    #[test]
    fn impl_block_input_is_rejected_at_parse_time() {
        let item = quote! {
            impl S {
                fn my_test(&self) -> anyhow::Result<()> { Ok(()) }
            }
        };
        let result = expand(well_formed_attribute(), item);

        assert!(result.is_err(), "impl block must fail");
    }

    #[test]
    fn function_without_return_type_is_rejected() {
        let item = quote! { fn my_test(fixture: &LlamaFixture<'_>) { } };
        let error = expand(well_formed_attribute(), item)
            .expect_err("missing return type must fail")
            .to_string();

        assert!(error.contains("anyhow::Result"), "got: {error}");
    }

    #[test]
    fn async_function_is_rejected() {
        let item = quote! {
            async fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let error = expand(well_formed_attribute(), item)
            .expect_err("async must fail")
            .to_string();

        assert!(error.contains("synchronous"), "got: {error}");
    }

    #[test]
    fn generic_function_is_rejected() {
        let item = quote! {
            fn my_test<T>(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let error = expand(well_formed_attribute(), item)
            .expect_err("generic must fail")
            .to_string();

        assert!(error.contains("generic"), "got: {error}");
    }

    #[test]
    fn malformed_item_token_stream_is_rejected() {
        let item = quote! { this is not a function };
        let result = expand(well_formed_attribute(), item);

        assert!(result.is_err(), "non-fn item must fail");
    }

    #[test]
    fn stacked_invocation_preserves_prior_submission() {
        let prior_layer_output = expand(well_formed_attribute(), well_formed_function())
            .expect("first layer must expand");

        let second_attribute = quote! {
            model_source = HuggingFace("second", "second.gguf"),
            n_gpu_layers = 1,
            use_mmap = false,
            use_mlock = false,
            n_ctx = 2,
            n_batch = 2,
            n_ubatch = 2
        };

        let second_layer_output = expand(second_attribute, prior_layer_output)
            .expect("stacked second invocation must expand")
            .to_string();

        assert!(
            second_layer_output.contains("\"my_test[bar.gguf]\""),
            "stacked output missing first trial name: {second_layer_output}",
        );
        assert!(
            second_layer_output.contains("\"my_test[second.gguf]\""),
            "stacked output missing second trial name: {second_layer_output}",
        );
        let occurrences = second_layer_output.matches("LlamaTestRegistration").count();
        assert!(
            occurrences >= 2,
            "stacked output should contain two LlamaTestRegistration submissions, found {occurrences}: {second_layer_output}",
        );
    }

    #[test]
    fn two_fn_definitions_in_input_are_rejected() {
        let item = quote! {
            fn first(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
            fn second(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let error = expand(well_formed_attribute(), item)
            .expect_err("two fns must fail")
            .to_string();

        assert!(error.contains("exactly one fn"), "got: {error}");
    }
}
