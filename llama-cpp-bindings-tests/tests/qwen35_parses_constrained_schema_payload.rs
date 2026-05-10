use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;
use serde_json::Value;
use serde_json::json;

const QWEN35_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const QWEN35_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

const NEGOTIATE_WITH_CAT_TOOLS_JSON: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "negotiate_with_cat",
            "description": "Attempt to negotiate with a cat. Outcomes are not guaranteed and may include the silent treatment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What you are trying to negotiate, e.g. 'get off the keyboard' or 'stop knocking things off the table'"
                    },
                    "bribe": {
                        "type": "string",
                        "enum": ["tuna", "salmon", "treats", "ear_scritches", "cardboard_box", "none"],
                        "description": "What you are offering in exchange"
                    },
                    "desperation_level": {
                        "type": "integer",
                        "description": "How desperate you are, on a scale from 1 (mildly annoyed human) to 10 (it is 3am)",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["topic"],
                "additionalProperties": false
            }
        }
    }
]"#;

const NEGOTIATE_WITH_CAT_INPUT: &str = "<tool_call>\n\
<function=negotiate_with_cat>\n\
<parameter=bribe>\n\
tuna\n\
</parameter>\n\
<parameter=desperation_level>\n\
8\n\
</parameter>\n\
<parameter=topic>\n\
get off the keyboard\n\
</parameter>\n\
</function>\n\
</tool_call>";

fn arguments_as_json(arguments: &ToolCallArguments) -> Result<&Value> {
    match arguments {
        ToolCallArguments::ValidJson(value) => Ok(value),
        ToolCallArguments::InvalidJson(raw) => {
            bail!("expected ValidJson arguments, got InvalidJson: {raw}")
        }
    }
}

#[test]
fn qwen35_parses_constrained_schema_payload() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(QWEN35_REPO, QWEN35_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let outcome = model.parse_chat_message(
        NEGOTIATE_WITH_CAT_TOOLS_JSON,
        NEGOTIATE_WITH_CAT_INPUT,
        false,
    )?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "Qwen 3.5's tool-call payload must be parsed by the wrapper-side duck-type pass; \
             got Unrecognized"
        );
    };

    assert_eq!(
        parsed.tool_calls.len(),
        1,
        "expected exactly one parsed tool call; got {:?}",
        parsed.tool_calls
    );
    assert_eq!(parsed.tool_calls[0].name, "negotiate_with_cat");
    assert_eq!(parsed.tool_calls[0].id, "call_0");
    assert_eq!(
        arguments_as_json(&parsed.tool_calls[0].arguments)?,
        &json!({
            "bribe": "tuna",
            "desperation_level": 8,
            "topic": "get off the keyboard",
        }),
    );

    Ok(())
}
