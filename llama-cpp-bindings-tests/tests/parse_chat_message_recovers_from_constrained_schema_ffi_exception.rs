use anyhow::Result;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings_tests::TestFixture;
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
            anyhow::bail!("expected ValidJson arguments, got InvalidJson: {raw}")
        }
    }
}

#[test]
fn recovers_negotiate_with_cat_when_constrained_schema_breaks_ffi_grammar() -> Result<()> {
    let fixture = TestFixture::shared();
    let model = fixture.default_model();

    let parsed = model.parse_chat_message(
        NEGOTIATE_WITH_CAT_TOOLS_JSON,
        NEGOTIATE_WITH_CAT_INPUT,
        false,
    )?;

    assert_eq!(
        parsed.tool_calls.len(),
        1,
        "expected exactly one recovered tool call; got {:?}",
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
