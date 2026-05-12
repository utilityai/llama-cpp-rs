use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const QWEN35_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const QWEN35_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

const TOOLS_JSON: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        }
    }
]"#;

const PLAIN_CONTENT: &str = "Sorry, I cannot help with that.";

#[test]
fn qwen35_recognizes_empty_tool_calls_when_input_is_plain_content_with_tools_requested()
-> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(QWEN35_REPO, QWEN35_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let outcome = model.parse_chat_message(TOOLS_JSON, PLAIN_CONTENT, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "Qwen 3.5 with tools requested + plain content must produce Recognized (with empty \
             tool_calls); got Unrecognized"
        );
    };
    assert!(
        parsed.tool_calls.is_empty(),
        "expected no tool calls; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}
