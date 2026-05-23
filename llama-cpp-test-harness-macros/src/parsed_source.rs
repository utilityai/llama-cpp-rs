use syn::Expr;
use syn::ExprCall;
use syn::ExprLit;
use syn::ExprPath;
use syn::Lit;

fn require_string_argument(expression: &Expr, field: &str, position: &str) -> syn::Result<String> {
    match expression {
        Expr::Lit(ExprLit {
            lit: Lit::Str(string_literal),
            ..
        }) => Ok(string_literal.value()),
        _ => Err(syn::Error::new_spanned(
            expression,
            format!("`{field}` argument `{position}` expects a string literal"),
        )),
    }
}

fn parse_huggingface_source(call: &ExprCall, field: &str) -> syn::Result<ParsedSource> {
    let args: Vec<&Expr> = call.args.iter().collect();
    let [repo_expr, file_expr] = args.as_slice() else {
        return Err(syn::Error::new_spanned(
            &call.args,
            format!(
                "`HuggingFace` expects exactly 2 string arguments (repo, file); got {got}",
                got = call.args.len()
            ),
        ));
    };
    let repo = require_string_argument(repo_expr, field, "HuggingFace.repo")?;
    let file = require_string_argument(file_expr, field, "HuggingFace.file")?;
    Ok(ParsedSource::HuggingFace { repo, file })
}

fn parse_local_path_source(call: &ExprCall, field: &str) -> syn::Result<ParsedSource> {
    let args: Vec<&Expr> = call.args.iter().collect();
    let [path_expr] = args.as_slice() else {
        return Err(syn::Error::new_spanned(
            &call.args,
            format!(
                "`LocalPath` expects exactly 1 string argument (path); got {got}",
                got = call.args.len()
            ),
        ));
    };
    let path = require_string_argument(path_expr, field, "LocalPath.path")?;
    Ok(ParsedSource::LocalPath(path))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParsedSource {
    HuggingFace { repo: String, file: String },
    LocalPath(String),
}

impl ParsedSource {
    pub fn display_suffix(&self) -> String {
        match self {
            Self::HuggingFace { file, .. } => file.clone(),
            Self::LocalPath(path) => std::path::Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .map_or_else(|| path.clone(), str::to_owned),
        }
    }

    pub fn parse(expression: &Expr, field: &str) -> syn::Result<Self> {
        let Expr::Call(call) = expression else {
            return Err(syn::Error::new_spanned(
                expression,
                format!("field `{field}` expects `HuggingFace(repo, file)` or `LocalPath(path)`"),
            ));
        };
        let Expr::Path(ExprPath { path, .. }) = call.func.as_ref() else {
            return Err(syn::Error::new_spanned(
                call.func.as_ref(),
                format!("field `{field}` expects `HuggingFace(...)` or `LocalPath(...)`"),
            ));
        };
        let variant_ident = path.get_ident().ok_or_else(|| {
            syn::Error::new_spanned(
                path,
                format!(
                    "field `{field}` expects the bare variant name `HuggingFace` or `LocalPath`"
                ),
            )
        })?;
        match variant_ident.to_string().as_str() {
            "HuggingFace" => parse_huggingface_source(call, field),
            "LocalPath" => parse_local_path_source(call, field),
            other => Err(syn::Error::new_spanned(
                variant_ident,
                format!("unknown source variant `{other}`; expected `HuggingFace` or `LocalPath`"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::ParsedSource;

    fn parse(source: &str) -> syn::Result<ParsedSource> {
        let expression: syn::Expr = parse_str(source)?;
        ParsedSource::parse(&expression, "model_source")
    }

    #[test]
    fn parses_huggingface_with_two_string_args() {
        let parsed = parse("HuggingFace(\"org/name\", \"file.gguf\")").expect("valid");

        assert_eq!(
            parsed,
            ParsedSource::HuggingFace {
                repo: "org/name".to_owned(),
                file: "file.gguf".to_owned(),
            },
        );
    }

    #[test]
    fn parses_local_path_with_one_string_arg() {
        let parsed = parse("LocalPath(\"/abs/local.gguf\")").expect("valid");

        assert_eq!(parsed, ParsedSource::LocalPath("/abs/local.gguf".to_owned()));
    }

    #[test]
    fn unknown_variant_is_rejected() {
        let message = parse("Mystery(\"a\", \"b\")")
            .expect_err("unknown variant must fail")
            .to_string();

        assert!(message.contains("unknown source variant"), "got: {message}");
    }

    #[test]
    fn non_call_expression_is_rejected() {
        let message = parse("\"plain\"")
            .expect_err("non-call must fail")
            .to_string();

        assert!(message.contains("HuggingFace"), "got: {message}");
        assert!(message.contains("LocalPath"), "got: {message}");
    }

    #[test]
    fn huggingface_with_wrong_arity_is_rejected() {
        let message = parse("HuggingFace(\"only-one\")")
            .expect_err("arity must fail")
            .to_string();

        assert!(message.contains("HuggingFace"), "got: {message}");
        assert!(message.contains("2 string"), "got: {message}");
    }

    #[test]
    fn local_path_with_wrong_arity_is_rejected() {
        let message = parse("LocalPath(\"a\", \"b\")")
            .expect_err("arity must fail")
            .to_string();

        assert!(message.contains("LocalPath"), "got: {message}");
        assert!(message.contains("1 string"), "got: {message}");
    }

    #[test]
    fn non_string_argument_is_rejected() {
        let message = parse("HuggingFace(42, \"file\")")
            .expect_err("non-string arg must fail")
            .to_string();

        assert!(message.contains("string literal"), "got: {message}");
    }

    #[test]
    fn huggingface_with_non_string_second_argument_is_rejected() {
        let message = parse("HuggingFace(\"repo\", 42)")
            .expect_err("non-string second arg must fail")
            .to_string();

        assert!(message.contains("string literal"), "got: {message}");
    }

    #[test]
    fn local_path_with_non_string_argument_is_rejected() {
        let message = parse("LocalPath(42)")
            .expect_err("non-string LocalPath arg must fail")
            .to_string();

        assert!(message.contains("string literal"), "got: {message}");
    }

    #[test]
    fn unparseable_input_returns_err() {
        let result = parse("@&^!");

        assert!(result.is_err(), "garbage input must fail to parse as syn::Expr");
    }

    #[test]
    fn non_path_function_expression_is_rejected() {
        let message = parse("(closure)(\"a\")")
            .expect_err("non-path func must fail")
            .to_string();

        assert!(message.contains("HuggingFace"), "got: {message}");
    }

    #[test]
    fn qualified_path_variant_is_rejected() {
        let message = parse("some::Other::HuggingFace(\"a\", \"b\")")
            .expect_err("qualified path must fail")
            .to_string();

        assert!(message.contains("bare variant name"), "got: {message}");
    }

    #[test]
    fn display_suffix_huggingface_returns_file() {
        let source = ParsedSource::HuggingFace {
            repo: "org/name".to_owned(),
            file: "model.gguf".to_owned(),
        };

        assert_eq!(source.display_suffix(), "model.gguf");
    }

    #[test]
    fn display_suffix_local_path_returns_basename() {
        let source = ParsedSource::LocalPath("/abs/dir/model.gguf".to_owned());

        assert_eq!(source.display_suffix(), "model.gguf");
    }

    #[test]
    fn display_suffix_local_path_without_file_name_returns_full_path() {
        let source = ParsedSource::LocalPath("/abs/dir/..".to_owned());

        assert_eq!(source.display_suffix(), "/abs/dir/..");
    }
}
