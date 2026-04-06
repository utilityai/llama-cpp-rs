# llama-cpp-rs-2

[utilityai]: https://utilityai.ca

A wrapper around the [llama-cpp](https://github.com/ggerganov/llama.cpp/) library for rust.

# Info

This is part of the project powering all the LLMs at [utilityai], it is tightly coupled llama.cpp and mimics its API as
closly as possible while being safe in order to stay up to date.

# Tool Calling

`llama-cpp-2` exposes the raw llama.cpp OpenAI-compatible tool-calling flow, so Rust callers can pass tool definitions into chat templates and get the generated grammar back.

```rust
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use serde_json::json;

let template = model.chat_template(None)?;

let tools_json = json!([
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch current weather by city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        }
    }
])
.to_string();

let messages_json = json!([
    {
        "role": "system",
        "content": "You are a tool caller."
    },
    {
        "role": "user",
        "content": "Fetch the weather in Paris."
    }
])
.to_string();

let params = OpenAIChatTemplateParams {
    messages_json: &messages_json,
    tools_json: Some(&tools_json),
    tool_choice: Some("auto"),
    json_schema: None,
    grammar: None,
    reasoning_format: None,
    chat_template_kwargs: Some("{}"),
    add_generation_prompt: true,
    use_jinja: true,
    parallel_tool_calls: false,
    enable_thinking: false,
    add_bos: false,
    add_eos: false,
    parse_tool_calls: true,
};

let result = model.apply_chat_template_oaicompat(&template, &params)?;
```

For standalone grammar generation from a JSON schema string, use `llama_cpp_2::json_schema_to_grammar`.

# Dependencies

This uses bindgen to build the bindings to llama.cpp. This means that you need to have clang installed on your system.

If this is a problem for you, open an issue, and we can look into including the bindings. 

See [bindgen](https://rust-lang.github.io/rust-bindgen/requirements.html) for more information.

# Disclaimer

This crate is *not safe*. There is absolutly ways to misuse the llama.cpp API provided to create UB, please create an issue if you spot one. Do not use this code for tasks where UB is not acceptable.

This is not a simple library to use. In an ideal world a nice abstraction would be written on top of this crate to
provide an ergonomic API - the benefits of this crate over raw bindings is safety (and not much of it as that) and not much else.

We compensate for this shortcoming (we hope) by providing lots of examples and good documentation. Testing is a work in
progress.

# Contributing

Contributions are welcome. Please open an issue before starting work on a non-trivial PR.
