use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;
use serde_json::Value;
use serde_json::json;

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_constrained_schema_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = fixture.model.parse_chat_message(
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

    assert_eq!(parsed.tool_calls.len(), 1);
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

llama_tests_main!();
