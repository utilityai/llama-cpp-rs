use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const DEEPSEEK_R1_8B_REPO: &str = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF";
const DEEPSEEK_R1_8B_FILE: &str = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";

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

const GEMMA_PAIRED_QUOTE_PAYLOAD: &str = "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";

#[test]
fn deepseek_r1_8b_duck_types_gemma_paired_quote() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(DEEPSEEK_R1_8B_REPO, DEEPSEEK_R1_8B_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let outcome = model.parse_chat_message(TOOLS_JSON, GEMMA_PAIRED_QUOTE_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "duck-type pass must recognise Gemma paired-quote on a model with no registered \
             template; got Unrecognized"
        );
    };
    assert_eq!(
        parsed.tool_calls.len(),
        1,
        "expected one tool call; got {:?}",
        parsed.tool_calls
    );
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    let location = match &parsed.tool_calls[0].arguments {
        ToolCallArguments::ValidJson(value) => value
            .get("location")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        ToolCallArguments::InvalidJson(raw) => {
            bail!("expected ValidJson, got InvalidJson: {raw}");
        }
    };
    assert_eq!(location.as_deref(), Some("Paris"));

    Ok(())
}
