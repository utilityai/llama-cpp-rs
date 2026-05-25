use std::collections::HashSet;

use proc_macro2::Ident;
use proc_macro2::Span;
use syn::Expr;
use syn::ExprLit;
use syn::Lit;
use syn::Meta;
use syn::Token;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::punctuated::Punctuated;

use crate::parsed_context_params::ParsedContextParams;
use crate::parsed_model_load_params::ParsedModelLoadParams;
use crate::parsed_source::ParsedSource;

const REQUIRED_FIELDS: &[&str] = &[
    "model_source",
    "n_gpu_layers",
    "use_mmap",
    "use_mlock",
    "n_ctx",
    "n_batch",
    "n_ubatch",
];

const OPTIONAL_FIELDS: &[&str] = &[
    "mmproj_source",
    "embeddings",
    "n_seq_max",
    "n_threads_batch",
    "void_logs",
];

fn literal_from_expression(expression: &Expr) -> syn::Result<&Lit> {
    if let Expr::Lit(ExprLit { lit, .. }) = expression {
        Ok(lit)
    } else {
        Err(syn::Error::new_spanned(
            expression,
            "expected a literal (string, integer, or bool)",
        ))
    }
}

fn require_int_lit(literal: &Lit, field: &str) -> syn::Result<u32> {
    if let Lit::Int(int_literal) = literal {
        int_literal.base10_parse::<u32>().map_err(|parse_error| {
            syn::Error::new_spanned(
                literal,
                format!(
                    "field `{field}` expects a non-negative integer that fits in u32: {parse_error}"
                ),
            )
        })
    } else {
        Err(syn::Error::new_spanned(
            literal,
            format!("field `{field}` expects an integer literal"),
        ))
    }
}

fn require_i32_lit(literal: &Lit, field: &str) -> syn::Result<i32> {
    if let Lit::Int(int_literal) = literal {
        int_literal.base10_parse::<i32>().map_err(|parse_error| {
            syn::Error::new_spanned(
                literal,
                format!("field `{field}` expects an integer that fits in i32: {parse_error}"),
            )
        })
    } else {
        Err(syn::Error::new_spanned(
            literal,
            format!("field `{field}` expects an integer literal"),
        ))
    }
}

fn require_bool_lit(literal: &Lit, field: &str) -> syn::Result<bool> {
    if let Lit::Bool(bool_literal) = literal {
        Ok(bool_literal.value())
    } else {
        Err(syn::Error::new_spanned(
            literal,
            format!("field `{field}` expects a bool literal (`true` or `false`)"),
        ))
    }
}

fn require<TValue>(value: Option<TValue>, field: &str, span: Span) -> syn::Result<TValue> {
    value.ok_or_else(|| syn::Error::new(span, format!("missing required field `{field}`")))
}

#[derive(Default)]
struct AttributeAccumulator {
    model_source: Option<ParsedSource>,
    mmproj_source: Option<ParsedSource>,
    n_gpu_layers: Option<u32>,
    use_mmap: Option<bool>,
    use_mlock: Option<bool>,
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    embeddings: Option<bool>,
    n_seq_max: Option<u32>,
    n_threads_batch: Option<i32>,
    void_logs: Option<bool>,
}

fn dispatch_field(
    accumulator: &mut AttributeAccumulator,
    identifier: &Ident,
    name: &str,
    value: &Expr,
) -> syn::Result<()> {
    match name {
        "model_source" => {
            accumulator.model_source = Some(ParsedSource::parse(value, "model_source")?);
        }
        "mmproj_source" => {
            accumulator.mmproj_source = Some(ParsedSource::parse(value, "mmproj_source")?);
        }
        "n_gpu_layers" => {
            accumulator.n_gpu_layers = Some(require_int_lit(
                literal_from_expression(value)?,
                "n_gpu_layers",
            )?);
        }
        "n_ctx" => {
            accumulator.n_ctx = Some(require_int_lit(literal_from_expression(value)?, "n_ctx")?);
        }
        "n_batch" => {
            accumulator.n_batch =
                Some(require_int_lit(literal_from_expression(value)?, "n_batch")?);
        }
        "n_ubatch" => {
            accumulator.n_ubatch = Some(require_int_lit(
                literal_from_expression(value)?,
                "n_ubatch",
            )?);
        }
        "use_mmap" => {
            accumulator.use_mmap = Some(require_bool_lit(
                literal_from_expression(value)?,
                "use_mmap",
            )?);
        }
        "use_mlock" => {
            accumulator.use_mlock = Some(require_bool_lit(
                literal_from_expression(value)?,
                "use_mlock",
            )?);
        }
        "embeddings" => {
            accumulator.embeddings = Some(require_bool_lit(
                literal_from_expression(value)?,
                "embeddings",
            )?);
        }
        "n_seq_max" => {
            accumulator.n_seq_max = Some(require_int_lit(
                literal_from_expression(value)?,
                "n_seq_max",
            )?);
        }
        "n_threads_batch" => {
            accumulator.n_threads_batch = Some(require_i32_lit(
                literal_from_expression(value)?,
                "n_threads_batch",
            )?);
        }
        "void_logs" => {
            accumulator.void_logs = Some(require_bool_lit(
                literal_from_expression(value)?,
                "void_logs",
            )?);
        }
        "repo" | "file" | "mmproj_file" => {
            return Err(syn::Error::new_spanned(
                identifier,
                format!(
                    "field `{name}` was removed; use `model_source = HuggingFace(repo, file)` or `model_source = LocalPath(path)` (and `mmproj_source` for mmproj)"
                ),
            ));
        }
        other => {
            return Err(syn::Error::new_spanned(
                identifier,
                format!(
                    "unknown field `{other}`; expected one of: {}, {}",
                    REQUIRED_FIELDS.join(", "),
                    OPTIONAL_FIELDS.join(", "),
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct ParsedArgs {
    pub model_source: ParsedSource,
    pub mmproj_source: Option<ParsedSource>,
    pub model_load_params: ParsedModelLoadParams,
    pub context_params: ParsedContextParams,
    pub void_logs: bool,
}

impl Parse for ParsedArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut seen: HashSet<String> = HashSet::new();
        let mut accumulator = AttributeAccumulator::default();

        for meta in metas {
            let Meta::NameValue(name_value) = meta else {
                return Err(syn::Error::new_spanned(
                    meta,
                    "expected `name = value` form",
                ));
            };
            let identifier = name_value.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(&name_value.path, "expected a simple identifier")
            })?;
            let name = identifier.to_string();

            if !seen.insert(name.clone()) {
                return Err(syn::Error::new_spanned(
                    identifier,
                    format!("duplicate field `{name}`"),
                ));
            }
            dispatch_field(&mut accumulator, identifier, &name, &name_value.value)?;
        }

        let span = Span::call_site();
        Ok(Self {
            model_source: require(accumulator.model_source, "model_source", span)?,
            mmproj_source: accumulator.mmproj_source,
            model_load_params: ParsedModelLoadParams {
                n_gpu_layers: require(accumulator.n_gpu_layers, "n_gpu_layers", span)?,
                use_mmap: require(accumulator.use_mmap, "use_mmap", span)?,
                use_mlock: require(accumulator.use_mlock, "use_mlock", span)?,
            },
            context_params: ParsedContextParams {
                n_ctx: require(accumulator.n_ctx, "n_ctx", span)?,
                n_batch: require(accumulator.n_batch, "n_batch", span)?,
                n_ubatch: require(accumulator.n_ubatch, "n_ubatch", span)?,
                n_seq_max: accumulator.n_seq_max.unwrap_or(1),
                n_threads_batch: accumulator.n_threads_batch,
                embeddings: accumulator.embeddings.unwrap_or(false),
            },
            void_logs: accumulator.void_logs.unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::ParsedArgs;
    use crate::parsed_source::ParsedSource;

    const ALL_REQUIRED: &str = "\
        model_source = HuggingFace(\"foo\", \"bar.gguf\"), \
        n_gpu_layers = 0, \
        use_mmap = true, \
        use_mlock = false, \
        n_ctx = 512, \
        n_batch = 128, \
        n_ubatch = 64";

    fn parse(source: &str) -> syn::Result<ParsedArgs> {
        parse_str(source)
    }

    #[test]
    fn parses_all_required_fields() {
        let parsed = parse(ALL_REQUIRED).expect("required-only must parse");

        assert_eq!(
            parsed.model_source,
            ParsedSource::HuggingFace {
                repo: "foo".to_owned(),
                file: "bar.gguf".to_owned(),
            },
        );
        assert_eq!(parsed.model_load_params.n_gpu_layers, 0);
        assert!(parsed.model_load_params.use_mmap);
        assert!(!parsed.model_load_params.use_mlock);
        assert_eq!(parsed.context_params.n_ctx, 512);
        assert_eq!(parsed.context_params.n_batch, 128);
        assert_eq!(parsed.context_params.n_ubatch, 64);
        assert!(parsed.mmproj_source.is_none());
    }

    #[test]
    fn parses_local_path_model_source() {
        let source = "\
            model_source = LocalPath(\"/abs/local/model.gguf\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let parsed = parse(source).expect("LocalPath must parse");

        assert_eq!(
            parsed.model_source,
            ParsedSource::LocalPath("/abs/local/model.gguf".to_owned()),
        );
    }

    #[test]
    fn parses_optional_mmproj_source_huggingface() {
        let source =
            format!("{ALL_REQUIRED}, mmproj_source = HuggingFace(\"foo\", \"mmproj-F16.gguf\")");
        let parsed = parse(&source).expect("with mmproj_source must parse");

        assert_eq!(
            parsed.mmproj_source,
            Some(ParsedSource::HuggingFace {
                repo: "foo".to_owned(),
                file: "mmproj-F16.gguf".to_owned(),
            }),
        );
    }

    #[test]
    fn parses_optional_mmproj_source_local_path() {
        let source = format!("{ALL_REQUIRED}, mmproj_source = LocalPath(\"/abs/mmproj.gguf\")");
        let parsed = parse(&source).expect("with mmproj_source LocalPath must parse");

        assert_eq!(
            parsed.mmproj_source,
            Some(ParsedSource::LocalPath("/abs/mmproj.gguf".to_owned())),
        );
    }

    #[test]
    fn legacy_repo_field_is_rejected_with_migration_hint() {
        let source = "repo = \"foo\", file = \"bar\", n_gpu_layers = 0, use_mmap = true, \
            use_mlock = false, n_ctx = 1, n_batch = 1, n_ubatch = 1";
        let message = parse(source)
            .expect_err("legacy repo must be rejected")
            .to_string();

        assert!(message.contains("model_source"), "got: {message}");
    }

    #[test]
    fn legacy_mmproj_file_field_is_rejected_with_migration_hint() {
        let source = format!("{ALL_REQUIRED}, mmproj_file = \"mmproj.gguf\"");
        let message = parse(&source)
            .expect_err("legacy mmproj_file must be rejected")
            .to_string();

        assert!(message.contains("mmproj_source"), "got: {message}");
    }

    #[test]
    fn missing_model_source_is_rejected() {
        let source = "n_gpu_layers = 0, use_mmap = true, use_mlock = false, \
            n_ctx = 1, n_batch = 1, n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing model_source must fail")
            .to_string();

        assert!(
            message.contains("missing required field `model_source`"),
            "got: {message}"
        );
    }

    #[test]
    fn missing_n_ctx_is_rejected() {
        let source = "model_source = HuggingFace(\"x\", \"y\"), n_gpu_layers = 0, use_mmap = true, \
            use_mlock = false, n_batch = 1, n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing n_ctx must fail")
            .to_string();

        assert!(
            message.contains("missing required field `n_ctx`"),
            "got: {message}"
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        let source = format!("{ALL_REQUIRED}, surprise = 1");
        let message = parse(&source)
            .expect_err("unknown field must fail")
            .to_string();

        assert!(
            message.contains("unknown field `surprise`"),
            "got: {message}"
        );
    }

    #[test]
    fn duplicate_field_is_rejected() {
        let source = format!("{ALL_REQUIRED}, model_source = HuggingFace(\"other\", \"o.gguf\")");
        let message = parse(&source).expect_err("duplicate must fail").to_string();

        assert!(
            message.contains("duplicate field `model_source`"),
            "got: {message}"
        );
    }

    #[test]
    fn non_name_value_form_is_rejected() {
        let source = "model_source, file = \"x\"";
        let message = parse(source).expect_err("bare ident must fail").to_string();

        assert!(message.contains("name = value"), "got: {message}");
    }

    #[test]
    fn non_literal_value_for_scalar_field_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = some_const, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("non-literal value must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn wrong_literal_kind_for_int_field_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = \"nine\", \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("string for int field must fail")
            .to_string();

        assert!(message.contains("integer literal"), "got: {message}");
    }

    #[test]
    fn wrong_literal_kind_for_bool_field_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mmap = 1, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("int for bool field must fail")
            .to_string();

        assert!(message.contains("bool literal"), "got: {message}");
    }

    #[test]
    fn negative_int_for_u32_field_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = -1, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("negative int must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn complex_path_field_name_is_rejected() {
        let source = "\
            foo::bar = 1, \
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("path field name must fail")
            .to_string();

        assert!(message.contains("simple identifier"), "got: {message}");
    }

    #[test]
    fn overflowing_int_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 99999999999, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source).expect_err("overflow must fail").to_string();

        assert!(message.contains("u32"), "got: {message}");
    }

    #[test]
    fn overflowing_i32_for_n_threads_batch_is_rejected() {
        let source = format!("{ALL_REQUIRED}, n_threads_batch = 99999999999");
        let message = parse(&source)
            .expect_err("i32 overflow must fail")
            .to_string();

        assert!(message.contains("i32"), "got: {message}");
    }

    #[test]
    fn missing_n_gpu_layers_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing n_gpu_layers must fail")
            .to_string();

        assert!(
            message.contains("missing required field `n_gpu_layers`"),
            "got: {message}"
        );
    }

    #[test]
    fn missing_use_mmap_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing use_mmap must fail")
            .to_string();

        assert!(
            message.contains("missing required field `use_mmap`"),
            "got: {message}"
        );
    }

    #[test]
    fn missing_use_mlock_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing use_mlock must fail")
            .to_string();

        assert!(
            message.contains("missing required field `use_mlock`"),
            "got: {message}"
        );
    }

    #[test]
    fn missing_n_batch_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("missing n_batch must fail")
            .to_string();

        assert!(
            message.contains("missing required field `n_batch`"),
            "got: {message}"
        );
    }

    #[test]
    fn missing_n_ubatch_is_rejected() {
        let source = "\
            model_source = HuggingFace(\"x\", \"y\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1";
        let message = parse(source)
            .expect_err("missing n_ubatch must fail")
            .to_string();

        assert!(
            message.contains("missing required field `n_ubatch`"),
            "got: {message}"
        );
    }

    #[test]
    fn optional_embeddings_defaults_to_false_when_absent() {
        let parsed = parse(ALL_REQUIRED).expect("required-only must parse");

        assert!(!parsed.context_params.embeddings);
    }

    #[test]
    fn optional_embeddings_true_is_parsed() {
        let source = format!("{ALL_REQUIRED}, embeddings = true");
        let parsed = parse(&source).expect("embeddings = true must parse");

        assert!(parsed.context_params.embeddings);
    }

    #[test]
    fn optional_embeddings_false_is_parsed() {
        let source = format!("{ALL_REQUIRED}, embeddings = false");
        let parsed = parse(&source).expect("embeddings = false must parse");

        assert!(!parsed.context_params.embeddings);
    }

    #[test]
    fn optional_embeddings_rejects_non_bool_literal() {
        let source = format!("{ALL_REQUIRED}, embeddings = 1");
        let message = parse(&source)
            .expect_err("embeddings with int must fail")
            .to_string();

        assert!(message.contains("bool literal"), "got: {message}");
    }

    #[test]
    fn optional_n_seq_max_defaults_to_one_when_absent() {
        let parsed = parse(ALL_REQUIRED).expect("required-only must parse");

        assert_eq!(parsed.context_params.n_seq_max, 1);
    }

    #[test]
    fn optional_n_seq_max_is_parsed() {
        let source = format!("{ALL_REQUIRED}, n_seq_max = 4");
        let parsed = parse(&source).expect("n_seq_max = 4 must parse");

        assert_eq!(parsed.context_params.n_seq_max, 4);
    }

    #[test]
    fn optional_n_threads_batch_defaults_to_none_when_absent() {
        let parsed = parse(ALL_REQUIRED).expect("required-only must parse");

        assert_eq!(parsed.context_params.n_threads_batch, None);
    }

    #[test]
    fn optional_n_threads_batch_is_parsed_when_positive() {
        let source = format!("{ALL_REQUIRED}, n_threads_batch = 8");
        let parsed = parse(&source).expect("n_threads_batch = 8 must parse");

        assert_eq!(parsed.context_params.n_threads_batch, Some(8));
    }

    #[test]
    fn optional_n_threads_batch_rejects_non_integer_literal() {
        let source = format!("{ALL_REQUIRED}, n_threads_batch = \"eight\"");
        let message = parse(&source)
            .expect_err("string for n_threads_batch must fail")
            .to_string();

        assert!(message.contains("integer literal"), "got: {message}");
    }

    #[test]
    fn optional_void_logs_defaults_to_false_when_absent() {
        let parsed = parse(ALL_REQUIRED).expect("required-only must parse");

        assert!(!parsed.void_logs);
    }

    #[test]
    fn optional_void_logs_true_is_parsed() {
        let source = format!("{ALL_REQUIRED}, void_logs = true");
        let parsed = parse(&source).expect("void_logs = true must parse");

        assert!(parsed.void_logs);
    }

    fn override_field(field: &str, replacement: &str) -> String {
        let parts: [(&str, &str); 7] = [
            ("model_source", "HuggingFace(\"foo\", \"bar.gguf\")"),
            ("n_gpu_layers", "0"),
            ("use_mmap", "true"),
            ("use_mlock", "false"),
            ("n_ctx", "512"),
            ("n_batch", "128"),
            ("n_ubatch", "64"),
        ];
        parts
            .iter()
            .map(|(name, value)| {
                let resolved = if *name == field { replacement } else { *value };
                format!("{name} = {resolved}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn append_field(field: &str, value: &str) -> String {
        format!("{ALL_REQUIRED}, {field} = {value}")
    }

    #[test]
    fn each_int_dispatch_arm_rejects_non_literal_value() {
        for field in ["n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"] {
            let source = override_field(field, "some_const");
            let message = parse(&source).expect_err(field).to_string();

            assert!(message.contains("literal"), "{field}: {message}");
        }
    }

    #[test]
    fn each_bool_dispatch_arm_rejects_non_literal_value() {
        for field in ["use_mmap", "use_mlock"] {
            let source = override_field(field, "some_const");
            let message = parse(&source).expect_err(field).to_string();

            assert!(message.contains("literal"), "{field}: {message}");
        }
    }

    #[test]
    fn each_int_dispatch_arm_rejects_wrong_literal_kind() {
        for field in ["n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"] {
            let source = override_field(field, "\"not-an-int\"");
            let message = parse(&source).expect_err(field).to_string();

            assert!(message.contains("integer literal"), "{field}: {message}");
        }
    }

    #[test]
    fn each_bool_dispatch_arm_rejects_wrong_literal_kind() {
        for field in ["use_mmap", "use_mlock"] {
            let source = override_field(field, "0");
            let message = parse(&source).expect_err(field).to_string();

            assert!(message.contains("bool literal"), "{field}: {message}");
        }
    }

    #[test]
    fn optional_n_seq_max_rejects_non_literal_value() {
        let source = append_field("n_seq_max", "some_const");
        let message = parse(&source)
            .expect_err("n_seq_max non-literal must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn optional_n_seq_max_rejects_wrong_literal_kind() {
        let source = append_field("n_seq_max", "\"four\"");
        let message = parse(&source)
            .expect_err("n_seq_max wrong-kind must fail")
            .to_string();

        assert!(message.contains("integer literal"), "got: {message}");
    }

    #[test]
    fn optional_n_threads_batch_rejects_non_literal_value() {
        let source = append_field("n_threads_batch", "some_const");
        let message = parse(&source)
            .expect_err("n_threads_batch non-literal must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn optional_embeddings_rejects_non_literal_value() {
        let source = append_field("embeddings", "some_const");
        let message = parse(&source)
            .expect_err("embeddings non-literal must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn optional_void_logs_rejects_non_literal_value() {
        let source = append_field("void_logs", "some_const");
        let message = parse(&source)
            .expect_err("void_logs non-literal must fail")
            .to_string();

        assert!(message.contains("literal"), "got: {message}");
    }

    #[test]
    fn optional_void_logs_rejects_wrong_literal_kind() {
        let source = append_field("void_logs", "1");
        let message = parse(&source)
            .expect_err("void_logs wrong-kind must fail")
            .to_string();

        assert!(message.contains("bool literal"), "got: {message}");
    }

    #[test]
    fn optional_mmproj_source_rejects_unknown_variant() {
        let source = append_field("mmproj_source", "Mystery(\"a\", \"b\")");
        let message = parse(&source)
            .expect_err("mmproj_source unknown variant must fail")
            .to_string();

        assert!(message.contains("unknown source variant"), "got: {message}");
    }

    #[test]
    fn model_source_with_unknown_variant_is_rejected() {
        let source = "\
            model_source = Mystery(\"a\", \"b\"), \
            n_gpu_layers = 0, \
            use_mmap = true, \
            use_mlock = false, \
            n_ctx = 1, \
            n_batch = 1, \
            n_ubatch = 1";
        let message = parse(source)
            .expect_err("model_source unknown variant must fail")
            .to_string();

        assert!(message.contains("unknown source variant"), "got: {message}");
    }

    #[test]
    fn unparseable_attribute_token_stream_is_rejected() {
        let result = parse("@&^!");

        assert!(
            result.is_err(),
            "garbage attribute body must fail to parse as Punctuated<Meta, ,>"
        );
    }
}
