use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_tests::gpu_backend::inference_model_params;
use llama_cpp_bindings_tests::gpu_backend::require_compiled_backends_present;
use llama_cpp_bindings_tests::test_model::download_file_from;

const DEEPSEEK_R1_8B_REPO: &str = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF";
const DEEPSEEK_R1_8B_FILE: &str = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";

const PLAIN_CONTENT: &str = "Hello there.";

#[test]
fn deepseek_r1_8b_recognizes_empty_tool_calls_when_tools_not_requested() -> Result<()> {
    let backend = LlamaBackend::init()?;
    require_compiled_backends_present()?;

    let path = download_file_from(DEEPSEEK_R1_8B_REPO, DEEPSEEK_R1_8B_FILE)?;
    let params = inference_model_params();
    let model = LlamaModel::load_from_file(&backend, &path, &params)?;

    let outcome = model.parse_chat_message("[]", PLAIN_CONTENT, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("plain content with empty tools array must produce Recognized; got Unrecognized");
    };
    assert!(
        parsed.tool_calls.is_empty(),
        "expected no tool calls; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}
