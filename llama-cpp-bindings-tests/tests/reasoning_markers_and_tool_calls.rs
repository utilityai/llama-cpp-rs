#![expect(
    clippy::unnecessary_wraps,
    reason = "trial fns share the harness LlamaTestFn signature even when their bodies never propagate"
)]

use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ToolCallArgsShape;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use llama_cpp_test_harness::llama_tests_main;
use serde_json::Value;
use serde_json::json;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn deepseek_r1_8b_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const DEEPSEEK_R1_8B_THINKING_DISABLED_PROMPT: &str = "\
<｜User｜>What is 2 + 2?<｜Assistant｜><think>

</think>

";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens =
        model.str_to_token(DEEPSEEK_R1_8B_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(
        !outcome.generated_raw.is_empty(),
        "DeepSeek-R1-8B: must generate at least one token"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "DeepSeek-R1-8B thinking-disabled: classifier must not emit any Reasoning token \
         when the prompt closes the think block before generation begins; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "DeepSeek-R1-8B thinking-disabled: prompt-token replay must move section to Content \
         before generation, so no Undeterminable tokens may be emitted; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "DeepSeek-R1-8B thinking-disabled: usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "DeepSeek-R1-8B thinking-disabled: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "DeepSeek-R1-8B thinking-disabled: classifier must emit at least one Content token"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "DeepSeek-R1-8B thinking-disabled: completion tokens must equal observed Content tokens"
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.content_stream.contains(forbidden),
            "DeepSeek-R1-8B thinking-disabled: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "test asserts many distinct properties of DeepSeek-R1-8B reasoning output; shortening messages or splitting the body would reduce diagnostic signal at failure time"
)]
#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn deepseek_r1_8b_classifier_emits_reasoning_for_thinking_enabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 1500;

    // DeepSeek-R1-Distill-Llama-8B uses `<think>...</think>` reasoning markers
    // and full-width-bar role tokens `<｜User｜>` / `<｜Assistant｜>` (U+FF5C,
    // not ASCII `|`). The chat template's `add_generation_prompt` ALWAYS appends
    // `<｜Assistant｜><think>\n` — DeepSeek-R1 is a pure reasoner with no
    // thinking-disabled mode — so the model resumes generation already inside
    // the reasoning block.
    const DEEPSEEK_R1_8B_THINKING_PROMPT: &str = "\
<｜User｜>What is 2 + 2?<｜Assistant｜><think>
";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(DEEPSEEK_R1_8B_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("DeepSeek-R1-8B chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(
        !outcome.generated_raw.is_empty(),
        "DeepSeek-R1-8B: must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "DeepSeek-R1-8B: classifier must emit at least one Reasoning token when the prompt \
         opens a <think> block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "DeepSeek-R1-8B: usage.reasoning_tokens must be non-zero when the prompt opens a \
         <think> block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "DeepSeek-R1-8B: prompt-token replay must move section to Reasoning before generation, \
         so no Undeterminable tokens may be emitted; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "DeepSeek-R1-8B: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "DeepSeek-R1-8B: completion tokens must equal observed Content + Reasoning"
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!(
            "DeepSeek-R1-8B didn't close its reasoning block within {MAX_GENERATED_TOKENS} \
             tokens — skipping strict parser-equality assertions"
        );
    } else {
        assert_eq!(
            outcome.reasoning_stream, parsed.reasoning_content,
            "DeepSeek-R1-8B: per-token reasoning stream must equal parser-side reasoning_content \
             (any difference means a marker leaked into the user-visible stream)",
        );
        assert_eq!(
            outcome.content_stream, parsed.content,
            "DeepSeek-R1-8B: per-token content stream must equal parser-side content \
             (any difference means a marker leaked into the user-visible stream)",
        );
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "DeepSeek-R1-8B: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "DeepSeek-R1-8B: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_gemma_paired_quote(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const GEMMA_PAIRED_QUOTE_PAYLOAD: &str =
        "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";

    let outcome =
        fixture
            .model
            .parse_chat_message(TOOLS_JSON, GEMMA_PAIRED_QUOTE_PAYLOAD, false)?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_glm_key_value_tags(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const GLM_KEY_VALUE_PAYLOAD: &str = "<tool_call>get_weather\
<arg_key>location</arg_key>\
<arg_value>Paris</arg_value>\
</tool_call>";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, GLM_KEY_VALUE_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "duck-type pass must recognise GLM key-value tags on a model with no registered \
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

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_mistral_bracketed_json(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const MISTRAL_BRACKETED_JSON_PAYLOAD: &str =
        r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#;

    let outcome =
        fixture
            .model
            .parse_chat_message(TOOLS_JSON, MISTRAL_BRACKETED_JSON_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "duck-type pass must recognise Mistral bracketed-JSON on a model with no registered \
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

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_qwen_xml(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const QWEN_XML_PAYLOAD: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, QWEN_XML_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "duck-type pass must recognise Qwen XML on a model with no registered template; \
             got Unrecognized"
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

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_recognizes_empty_tool_calls_when_input_is_plain_content_with_tools_requested(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
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

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, PLAIN_CONTENT, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "plain content with tools requested must produce Recognized (with empty tool_calls); \
             got Unrecognized"
        );
    };
    assert!(
        parsed.tool_calls.is_empty(),
        "expected no tool calls; got {:?}",
        parsed.tool_calls
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_recognizes_empty_tool_calls_when_tools_not_requested(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const PLAIN_CONTENT: &str = "Hello there.";

    let outcome = fixture
        .model
        .parse_chat_message("[]", PLAIN_CONTENT, false)?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn gemma4_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const GEMMA4_THINKING_DISABLED_PROMPT: &str = "\
<bos><start_of_turn>user\nReply with the single word: four. Do not explain.<end_of_turn>\n\
<start_of_turn>model\n<|channel>thought\n<channel|>\n";

    const FORBIDDEN_MARKERS: &[&str] = &["<|channel>thought", "<channel|>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GEMMA4_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(
        !outcome.generated_raw.is_empty(),
        "Gemma 4 must generate at least one token"
    );
    assert_eq!(
        outcome.observed_reasoning, 0,
        "Gemma 4 thinking-disabled: classifier must not emit any Reasoning token \
         when the prompt closes the thought channel before generation begins; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Gemma 4 thinking-disabled: prompt-token replay must move section to Content \
         before generation, so no Undeterminable tokens may be emitted; \
         generated={:?}",
        outcome.generated_raw
    );
    assert_eq!(
        usage.reasoning_tokens, 0,
        "Gemma 4 thinking-disabled: usage.reasoning_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Gemma 4 thinking-disabled: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert!(
        outcome.observed_content > 0,
        "Gemma 4 thinking-disabled: classifier must emit at least one Content token"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content,
        "Gemma 4 thinking-disabled: completion tokens must equal observed Content tokens"
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Gemma 4 thinking-disabled: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn gemma4_classifier_emits_reasoning_for_thinking_prompt(fixture: &LlamaFixture<'_>) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 1500;

    const GEMMA4_THINKING_PROMPT: &str = "\
<bos><start_of_turn>user\nReply with the single word: four. Do not explain.<end_of_turn>\n\
<start_of_turn>model\n<|channel>thought\n";

    const FORBIDDEN_MARKERS: &[&str] = &["<|channel>thought", "<channel|>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GEMMA4_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Gemma 4 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(
        !outcome.generated_raw.is_empty(),
        "Gemma 4 must generate at least one token"
    );
    assert!(
        outcome.observed_reasoning > 0,
        "Gemma 4 classifier must emit at least one Reasoning token when the model \
         emits a `<|channel>thought` block; outcome={outcome:?}",
    );
    assert!(
        usage.reasoning_tokens > 0,
        "Gemma 4 usage.reasoning_tokens must be non-zero when the model emits a \
         reasoning block; usage was {usage:?}"
    );
    assert_eq!(
        outcome.observed_undeterminable, 0,
        "Gemma 4: classifier must not emit Undeterminable when the model emits a \
         detected `<|channel>thought` marker; outcome={outcome:?}"
    );
    assert_eq!(
        usage.undeterminable_tokens, 0,
        "Gemma 4: usage.undeterminable_tokens must be zero; usage={usage:?}"
    );
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
        "Gemma 4: completion tokens must equal observed Content + Reasoning"
    );
    assert!(
        !parsed.reasoning_content.is_empty(),
        "Gemma 4 must close its reasoning block within {MAX_GENERATED_TOKENS} tokens; \
         increase the budget or pick a more direct prompt. generated={:?}",
        outcome.generated_raw,
    );

    for forbidden in FORBIDDEN_MARKERS {
        assert!(
            !outcome.reasoning_stream.contains(forbidden),
            "Gemma 4: reasoning_stream leaked marker {forbidden:?}; \
             reasoning_stream={:?}",
            outcome.reasoning_stream
        );
        assert!(
            !outcome.content_stream.contains(forbidden),
            "Gemma 4: content_stream leaked marker {forbidden:?}; \
             content_stream={:?}",
            outcome.content_stream
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma4_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const GEMMA4_PAIRED_QUOTE_PAYLOAD: &str =
        "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";

    let outcome =
        fixture
            .model
            .parse_chat_message(TOOLS_JSON, GEMMA4_PAIRED_QUOTE_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for Gemma 4 PairedQuote on a Gemma-4 model; got Unrecognized");
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

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma4_template_override_returns_full_markers(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let template = model
        .chat_template(None)
        .expect("Gemma 4 chat template must be present");
    let template_str = template.to_str().expect("template must be valid UTF-8");
    assert!(
        template_str.contains("<|tool_call>call:"),
        "Gemma 4 chat template must contain '<|tool_call>call:' fingerprint; \
         template starts with: {:?}",
        &template_str[..template_str.len().min(200)],
    );

    let markers = model
        .tool_call_markers()
        .expect("Gemma 4 must produce ToolCallMarkers via override registry");

    assert_eq!(markers.open, "<|tool_call>call:");
    assert_eq!(markers.close, "}");
    let ToolCallArgsShape::PairedQuote(shape) = markers.args_shape else {
        panic!("expected PairedQuote variant, got {:?}", markers.args_shape);
    };
    assert_eq!(shape.name_args_separator, "{");
    assert_eq!(shape.value_quote.open, "<|\"|>");
    assert_eq!(shape.value_quote.close, "<|\"|>");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn glm47_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const GLM47_THINKING_DISABLED_PROMPT: &str = "\
<|user|>
What is 2 + 2?
<|assistant|>
</think>

";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GLM47_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(!outcome.generated_raw.is_empty());
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(usage.completion_tokens(), outcome.observed_content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn glm47_classifier_emits_reasoning_for_thinking_enabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 1500;

    const GLM47_THINKING_PROMPT: &str = "\
<|user|>
What is 2 + 2?
<|assistant|>
<think>
";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(GLM47_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("GLM-4.7 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(usage.reasoning_tokens > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!(
            "GLM-4.7 didn't close its reasoning block within {MAX_GENERATED_TOKENS} tokens — \
             skipping strict parser-equality assertions"
        );
    } else {
        assert_eq!(outcome.reasoning_stream, parsed.reasoning_content);
        assert_eq!(outcome.content_stream, parsed.content);
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.reasoning_stream.contains(forbidden));
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn glm47_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const GLM47_KEY_VALUE_PAYLOAD: &str = "<tool_call>get_weather\
<arg_key>location</arg_key>\
<arg_value>Paris</arg_value>\
</tool_call>";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, GLM47_KEY_VALUE_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "expected Recognized for GLM-4.7 key-value tags on a GLM-4.7-Flash model; got Unrecognized"
        );
    };
    assert_eq!(parsed.tool_calls.len(), 1);
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

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn glm47_template_override_returns_full_markers(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let template = model
        .chat_template(None)
        .expect("GLM-4.7 chat template must be present");
    let template_str = template.to_str().expect("template must be valid UTF-8");
    assert!(template_str.contains("<arg_key>"));

    let markers = model
        .tool_call_markers()
        .expect("GLM-4.7 must produce ToolCallMarkers via override registry");

    assert_eq!(markers.open, "<tool_call>");
    assert_eq!(markers.close, "</tool_call>");
    let ToolCallArgsShape::KeyValueXmlTags(shape) = markers.args_shape else {
        panic!(
            "expected KeyValueXmlTags variant, got {:?}",
            markers.args_shape
        );
    };
    assert_eq!(shape.key_open, "<arg_key>");
    assert_eq!(shape.key_close, "</arg_key>");
    assert_eq!(shape.value_open, "<arg_value>");
    assert_eq!(shape.value_close, "</arg_value>");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn mistral3_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const MISTRAL3_THINKING_DISABLED_PROMPT: &str = "\
[INST]Reply with the single word: four. Do not explain.[/INST][THINK][/THINK]";

    const FORBIDDEN_MARKERS: &[&str] = &["[THINK]", "[/THINK]"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(MISTRAL3_THINKING_DISABLED_PROMPT, AddBos::Always)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(!outcome.generated_raw.is_empty());
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(usage.completion_tokens(), outcome.observed_content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn mistral3_classifier_emits_reasoning_for_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 768;

    const MISTRAL3_THINKING_PROMPT: &str = "\
[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\n\
First draft your thinking process (inner monologue) until you arrive at a response. \
Format your response using Markdown, and use LaTeX for any mathematical equations. \
Write both your thoughts and the response in the same language as the input.\n\n\
Your thinking process must follow the template below:\
[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. \
Be as casual and as long as you want until you are confident to generate the response \
to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT]\
[INST]Reply with the single word: four. Do not explain.[/INST]";

    const FORBIDDEN_MARKERS: &[&str] = &["[THINK]", "[/THINK]"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(MISTRAL3_THINKING_PROMPT, AddBos::Always)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Mistral 3 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(usage.reasoning_tokens > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
    );
    assert!(!parsed.reasoning_content.is_empty());
    assert_eq!(outcome.reasoning_stream, parsed.reasoning_content);
    assert_eq!(outcome.content_stream, parsed.content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.reasoning_stream.contains(forbidden));
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn mistral3_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const MISTRAL3_BRACKETED_JSON_PAYLOAD: &str =
        r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#;

    let outcome =
        fixture
            .model
            .parse_chat_message(TOOLS_JSON, MISTRAL3_BRACKETED_JSON_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "expected Recognized for Mistral 3 BracketedJson on a Mistral-3 model; got Unrecognized"
        );
    };
    assert_eq!(parsed.tool_calls.len(), 1);
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen35_chat_inference_emits_reasoning_when_template_auto_opens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_owned(),
        "Hello! How are you?".to_owned(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let mut classifier = model.sampled_token_classifier();
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 1024,
    }
    .run()?;

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(outcome.observed_tool_call, 0);

    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.5 chat template must be recognised by the parser; got Unrecognized");
    };
    assert!(!parsed.content.is_empty());

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, prompt_token_count);
    assert_eq!(usage.reasoning_tokens, outcome.observed_reasoning);
    assert_eq!(usage.undeterminable_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen35_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const QWEN35_THINKING_DISABLED_PROMPT: &str = "\
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>

</think>

";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(QWEN35_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(!outcome.generated_raw.is_empty());
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(usage.completion_tokens(), outcome.observed_content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen35_classifier_emits_reasoning_for_thinking_enabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 1500;

    const QWEN35_THINKING_PROMPT: &str = "\
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>
";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(QWEN35_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.5 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(usage.reasoning_tokens > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!(
            "Qwen3.5 didn't close its reasoning block within {MAX_GENERATED_TOKENS} tokens — \
             skipping strict parser-equality assertions"
        );
    } else {
        assert_eq!(outcome.reasoning_stream, parsed.reasoning_content);
        assert_eq!(outcome.content_stream, parsed.content);
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.reasoning_stream.contains(forbidden));
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_tool_call_payload(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const QWEN_XML_PAYLOAD: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, QWEN_XML_PAYLOAD, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for Qwen XML on a Qwen-3.5 model; got Unrecognized");
    };
    assert_eq!(parsed.tool_calls.len(), 1);
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_partial_tool_call_returns_pending_state(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const PARTIAL_QWEN_XML_PAYLOAD: &str = "<tool_call>\n<function=get_weather>\n<parameter=lo";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, PARTIAL_QWEN_XML_PAYLOAD, true)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for partial Qwen XML on a Qwen-3.5 model; got Unrecognized");
    };
    assert!(parsed.tool_calls.is_empty() || parsed.tool_calls.len() == 1);

    Ok(())
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
fn qwen35_parses_multiple_tool_calls(fixture: &LlamaFixture<'_>) -> Result<()> {
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

    const TWO_QWEN_XML_PAYLOADS: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>\n\
<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Berlin\n\
</parameter>\n\
</function>\n\
</tool_call>";

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, TWO_QWEN_XML_PAYLOADS, false)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!(
            "expected Recognized for two Qwen XML payloads on a Qwen-3.5 model; got Unrecognized"
        );
    };
    assert!(
        !parsed.tool_calls.is_empty(),
        "expected at least one tool call; got {:?}",
        parsed.tool_calls
    );

    Ok(())
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
fn qwen35_recognizes_empty_tool_calls_when_input_is_plain_content_with_tools_requested(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
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

    let outcome = fixture
        .model
        .parse_chat_message(TOOLS_JSON, PLAIN_CONTENT, false)?;

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen36_chat_inference_emits_reasoning_when_template_auto_opens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_owned(),
        "Hello! How are you?".to_owned(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let mut classifier = model.sampled_token_classifier();
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(512, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy();
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 1024,
    }
    .run()?;

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(outcome.observed_tool_call, 0);

    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, false)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.6 chat template must be recognised by the parser; got Unrecognized");
    };
    assert!(!parsed.content.is_empty());

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, prompt_token_count);
    assert_eq!(usage.reasoning_tokens, outcome.observed_reasoning);
    assert_eq!(usage.undeterminable_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen36_classifier_does_not_emit_reasoning_for_thinking_disabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    const QWEN36_THINKING_DISABLED_PROMPT: &str = "\
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>

</think>

";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(QWEN36_THINKING_DISABLED_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();

    assert!(!outcome.generated_raw.is_empty());
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert!(outcome.observed_content > 0);
    assert_eq!(usage.completion_tokens(), outcome.observed_content);

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    use_mmap = true,
    use_mlock = false,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen36_classifier_emits_reasoning_for_thinking_enabled_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 1500;

    const QWEN36_THINKING_PROMPT: &str = "\
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>
";

    const FORBIDDEN_MARKERS: &[&str] = &["<think>", "</think>"];

    let model = fixture.model;
    let backend = fixture.backend;

    let mut classifier = model.sampled_token_classifier();
    let prompt_tokens = model.str_to_token(QWEN36_THINKING_PROMPT, AddBos::Never)?;
    let prompt_token_count = u64::try_from(prompt_tokens.len())?;

    let mut batch = LlamaBatch::new(2048, 1)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    context.decode(&mut batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(0x00C0_FFEE),
    ]);
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        model,
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    let usage = classifier.usage();
    let parse_outcome = model.parse_chat_message("[]", &outcome.generated_raw, true)?;
    let ChatMessageParseOutcome::Recognized(parsed) = parse_outcome else {
        bail!("Qwen3.6 chat template must be recognised by the parser; got Unrecognized");
    };

    assert!(!outcome.generated_raw.is_empty());
    assert!(outcome.observed_reasoning > 0);
    assert!(usage.reasoning_tokens > 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(usage.undeterminable_tokens, 0);
    assert_eq!(
        usage.completion_tokens(),
        outcome.observed_content + outcome.observed_reasoning,
    );

    if parsed.reasoning_content.is_empty() {
        eprintln!("Qwen3.6 parser returned empty reasoning_content — relying on FORBIDDEN_MARKERS");
    } else {
        assert_eq!(outcome.reasoning_stream, parsed.reasoning_content);
        assert_eq!(outcome.content_stream, parsed.content);
    }

    for forbidden in FORBIDDEN_MARKERS {
        assert!(!outcome.reasoning_stream.contains(forbidden));
        assert!(!outcome.content_stream.contains(forbidden));
    }

    Ok(())
}
llama_tests_main!();
