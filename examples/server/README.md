# OpenAI Compatable Server Example

Minimal OpenAI-style chat completions server with support for:
- OpenAI-style messages
- Function calling (tools)
- JSON schema response formats

Run the server (add `--features cuda` if you have a CUDA GPU):

```bash
cargo run -p openai-server -- hf-model QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q8_0.gguf
```

Example request:

```sh
curl --location 'localhost:8080/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "dummy-model",
    "messages": [
      {
        "role": "system",
        "content": "You are an helpful assistant Tess"
      },
      {
        "role": "user",
        "content": "What is your name?"
      }
    ]
  }'
```

Example response:

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hi there! My name is Tess, and I'm here to assist you with any questions or tasks you may have. I'm a helpful assistant, and I'm always happy to lend a hand. What can I help you with today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1769269333,
  "id": "chatcmpl-1769269333",
  "model": "dummy-model",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 48,
    "prompt_tokens": 26,
    "total_tokens": 74
  }
}
```
