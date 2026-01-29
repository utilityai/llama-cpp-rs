# Interactive CLI Example

An interactive command-line chat application that demonstrates using the llama.cpp server infrastructure directly in-process from Rust.

## Overview

This example provides a chat interface similar to the upstream `cli.cpp` but uses the server infrastructure without HTTP. It shows how to:

- Initialize and configure the llama server context
- Run the server processing loop in a background thread
- Submit completion tasks and stream responses
- Maintain conversation history
- Handle user interrupts gracefully

## Architecture

### Main Components

- **`Args`** - Command-line argument parser (clap-based)
- **`CliContext`** - Manages server context and conversation state
- **`ChatMessage`** - Represents conversation messages with role and content

### Multi-threaded Design

The application uses a **two-thread architecture**:

1. **Background Server Thread** - Continuously processes tasks from the queue
2. **Main Thread** - Handles user interaction and displays results

```
┌─────────────────┐         ┌──────────────────────┐
│  Main Thread    │         │  Background Thread   │
│  (Interactive)  │         │   (Server Loop)      │
├─────────────────┤         ├──────────────────────┤
│ Read user input │         │ ctx_server           │
│ Post task ──────┼────────>│  .start_loop()       │
│ Poll results    │<────────┼─ Process tasks       │
│ Display tokens  │         │ Generate responses   │
└─────────────────┘         └──────────────────────┘
```

## Implementation Details

### Initialization Flow

Located in `main()` function (lines 296-335):

```rust
// 1. Parse command-line arguments
let args = Args::parse();

// 2. Get model path (local file or HuggingFace download)
let model_path = get_model_path(&args)?;

// 3. Initialize llama backend
let _backend = LlamaBackend::init()?;

// 4. Create server context
let ctx_server = ServerContext::new()?;

// 5. Load model into server
ctx_server.load_model(&model_path, model_params)?;

// 6. Start server loop in background thread
let ctx_for_loop = ctx_server.clone();
let server_thread = thread::spawn(move || {
    ctx_for_loop.start_loop();
});
```

### Task Submission and Response Streaming

The `generate_completion()` method (lines 175-260) handles the request/response cycle:

```rust
// 1. Get response reader from server context
let mut reader = self.ctx_server.get_response_reader()?;

// 2. Create unique task ID
let task_id = reader.get_new_id();

// 3. Convert conversation to JSON format
let messages_json = messages_to_json(&self.messages);

// 4. Post completion task to server
reader.post_completion(task_id, &self.default_params, &messages_json, &[])?;

// 5. Poll for streaming results
loop {
    let result = reader.next(|| is_interrupted.load(Ordering::Relaxed))?;
    
    match result.result_type() {
        TaskResultType::Partial => {
            // Stream tokens as they're generated
            for diff in result.get_diffs() {
                if let Some(delta) = &diff.content_delta {
                    print!("{}", delta);
                }
            }
        }
        TaskResultType::Final => {
            // Complete response received
            break;
        }
        _ => {}
    }
}
```

### Interactive Loop

The main loop (lines 403-454) provides the chat interface:

1. Read user input from stdin
2. Handle commands (`/exit`, `/regen`, `/clear`)
3. Add user message to conversation history
4. Generate and stream assistant response
5. Add assistant message to history
6. Display optional timing statistics

### Interrupt Handling

Ctrl+C behavior (lines 349-359):

- **First press**: Interrupts current generation
- **Second press**: Force terminates the application

```rust
ctrlc::set_handler(move || {
    if is_interrupted.load(Ordering::Relaxed) {
        // Second Ctrl+C - terminate immediately
        ctx_for_signal.terminate();
        std::process::exit(130);
    }
    // First Ctrl+C - set interrupt flag
    is_interrupted.store(true, Ordering::Relaxed);
})?;
```

## Usage

### Basic Usage

```bash
# Load a local model
cargo run --release -p cli -- --model path/to/model.gguf

# Or use the file subcommand
cargo run --release -p cli -- file path/to/model.gguf
```

### Download from HuggingFace

```bash
cargo run --release -p cli -- hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

### Advanced Options

```bash
cargo run --release -p cli -- \
  --model path/to/model.gguf \
  --ctx-size 4096 \              # Context window size
  --n-gpu-layers 35 \            # GPU offload layers
  --threads 8 \                  # CPU threads for generation
  --temperature 0.7 \            # Sampling temperature
  --top-p 0.95 \                 # Nucleus sampling
  --repeat-penalty 1.1 \         # Repetition penalty
  --flash-attn \                 # Enable flash attention
  --show-timings \               # Display performance stats
  --log-level 3 \                # Log verbosity (0-4)
  --system-prompt "You are a helpful assistant"
```

### Log Levels

| Level | Name   | Description |
|-------|--------|-------------|
| 0     | Output | No logs, output only |
| 1     | Error  | Error messages only (default) |
| 2     | Warn   | Warnings and errors |
| 3     | Info   | Info, warnings, and errors |
| 4     | Debug  | Most verbose (debug level) |

## Available Commands

While in the interactive chat:

- **`/exit`** or **Ctrl+C** - Stop or exit the application
- **`/regen`** - Regenerate the last assistant response
- **`/clear`** - Clear the entire chat history

## Features

- ✅ **Streaming output** - Tokens displayed as they're generated
- ✅ **Conversation memory** - Full message history maintained
- ✅ **Interrupt support** - Gracefully stop generation mid-stream
- ✅ **Reasoning tokens** - Supports thinking/reasoning mode (o1-style models)
- ✅ **Performance metrics** - Optional timing statistics
- ✅ **HuggingFace integration** - Automatic model downloads
- ✅ **Multimodal support** - Detects vision and audio capabilities
- ✅ **Log level control** - Configurable llama.cpp verbosity (0-4)

## Code References

| Feature | Location |
|---------|----------|
| Background thread creation | [main.rs#L332-L335](src/main.rs#L332-L335) |
| Task submission | [main.rs#L175-L181](src/main.rs#L175-L181) |
| Result streaming | [main.rs#L184-L232](src/main.rs#L184-L232) |
| Interactive loop | [main.rs#L403-L454](src/main.rs#L403-L454) |
| Interrupt handling | [main.rs#L349-L359](src/main.rs#L349-L359) |
| Cleanup/shutdown | [main.rs#L456-L458](src/main.rs#L456-L458) |

## See Also

- [Server module documentation](../../llama-cpp-2/src/server.rs)
- [Simple example](../simple/) - Basic single-shot completion
- [Embeddings example](../embeddings/) - Text embeddings generation
- [Reranker example](../reranker/) - Document reranking
