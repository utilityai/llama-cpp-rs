//! Interactive CLI for llama.cpp using the server infrastructure.
//!
//! This example demonstrates how to use the llama-server infrastructure from Rust
//! for an interactive chat experience, similar to the upstream cli.cpp.
//!
//! Usage:
//!   cargo run --release -p cli -- --model path/to/model.gguf
//!   cargo run --release -p cli -- hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::server::{
    ResultTimings, ServerContext, ServerModelParams, ServerTaskParams, TaskResultType,
};

const LLAMA_ASCII_LOGO: &str = r#"
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
"#;

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactive CLI for llama.cpp")]
struct Args {
    #[command(subcommand)]
    source: Option<ModelSource>,

    /// Path to the model file (alternative to subcommand)
    #[arg(short, long)]
    model: Option<String>,

    /// Context size (0 = use model default)
    #[arg(short = 'c', long, default_value = "4096")]
    ctx_size: i32,

    /// Number of layers to offload to GPU
    #[arg(short = 'g', long, default_value = "0")]
    n_gpu_layers: i32,

    /// Number of threads for generation
    #[arg(short = 't', long, default_value = "-1")]
    threads: i32,

    /// System prompt
    #[arg(short = 's', long)]
    system_prompt: Option<String>,

    /// Maximum tokens to generate (-1 = unlimited)
    #[arg(short = 'n', long, default_value = "-1")]
    n_predict: i32,

    /// Temperature for sampling
    #[arg(long, default_value = "0.8")]
    temperature: f32,

    /// Top-p (nucleus) sampling
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// Use flash attention
    #[arg(long)]
    flash_attn: bool,

    /// Show timing information after each response
    #[arg(long)]
    show_timings: bool,
}

#[derive(Subcommand, Debug)]
enum ModelSource {
    /// Load model from a local file path
    File {
        /// Path to the model file
        path: String,
    },
    /// Download and use a model from Hugging Face Hub
    HfModel {
        /// Hugging Face repository (e.g., "TheBloke/Llama-2-7B-GGUF")
        repo: String,
        /// Model filename within the repository
        file: String,
    },
}

/// Chat message for conversation history
#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// Convert messages to JSON array string
fn messages_to_json(messages: &[ChatMessage]) -> String {
    serde_json::to_string(messages).unwrap_or_else(|_| "[]".to_string())
}

/// CLI context holding server and conversation state
struct CliContext {
    ctx_server: ServerContext,
    messages: Vec<ChatMessage>,
    default_params: ServerTaskParams,
    is_interrupted: Arc<AtomicBool>,
}

impl CliContext {
    fn new(
        ctx_server: ServerContext,
        system_prompt: Option<String>,
        n_predict: i32,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
    ) -> Self {
        let mut messages = Vec::new();
        if let Some(prompt) = system_prompt {
            messages.push(ChatMessage::new("system", &prompt));
        }

        let mut default_params = ServerTaskParams::default();
        default_params.n_predict = n_predict;
        default_params.temperature = temperature;
        default_params.top_p = top_p;
        default_params.repeat_penalty = repeat_penalty;
        default_params.stream = true;
        default_params.timings_per_token = true;

        Self {
            ctx_server,
            messages,
            default_params,
            is_interrupted: Arc::new(AtomicBool::new(false)),
        }
    }

    fn generate_completion(&mut self) -> Result<(String, ResultTimings)> {
        let mut reader = self
            .ctx_server
            .get_response_reader()
            .context("Failed to get response reader")?;

        let task_id = reader.get_new_id();
        let messages_json = messages_to_json(&self.messages);

        reader
            .post_completion(task_id, &self.default_params, &messages_json, &[])
            .context("Failed to post completion task")?;

        let mut content = String::new();
        let mut timings = ResultTimings::default();
        let mut is_thinking = false;

        // Wait for results
        loop {
            // Clone the Arc for each iteration so the closure can own it
            let is_interrupted = self.is_interrupted.clone();
            let result = reader.next(move || is_interrupted.load(Ordering::Relaxed))?;
            
            let Some(result) = result else {
                break;
            };
            
            if self.is_interrupted.load(Ordering::Relaxed) {
                break;
            }

            if result.is_error() {
                if let Some(err) = result.get_error() {
                    eprintln!("\nError: {}", err);
                }
                break;
            }

            match result.result_type() {
                TaskResultType::Partial => {
                    timings = result.get_timings();
                    for diff in result.get_diffs() {
                        if let Some(delta) = &diff.content_delta {
                            if is_thinking {
                                println!("\n[End thinking]\n");
                                is_thinking = false;
                            }
                            content.push_str(delta);
                            print!("{}", delta);
                            io::stdout().flush()?;
                        }
                        if let Some(reasoning) = &diff.reasoning_content_delta {
                            if !is_thinking {
                                println!("[Start thinking]");
                                is_thinking = true;
                            }
                            print!("{}", reasoning);
                            io::stdout().flush()?;
                        }
                    }
                }
                TaskResultType::Final => {
                    timings = result.get_timings();
                    if let Some(final_content) = result.get_content() {
                        if content.is_empty() {
                            content = final_content;
                            print!("{}", content);
                            io::stdout().flush()?;
                        }
                    }
                    break;
                }
                TaskResultType::Error => {
                    if let Some(err) = result.get_error() {
                        eprintln!("\nError: {}", err);
                    }
                    break;
                }
                _ => {}
            }
        }

        // Reset interrupt flag
        self.is_interrupted.store(false, Ordering::Relaxed);

        Ok((content, timings))
    }
}

fn get_model_path(args: &Args) -> Result<String> {
    // First check if --model was provided directly
    if let Some(model_path) = &args.model {
        return Ok(model_path.clone());
    }

    // Otherwise use the subcommand
    match &args.source {
        Some(ModelSource::File { path }) => Ok(path.clone()),
        Some(ModelSource::HfModel { repo, file }) => {
            println!("Downloading model from Hugging Face: {}/{}", repo, file);
            let api = hf_hub::api::sync::Api::new()?;
            let model = api.model(repo.clone());
            let path = model.get(file)?;
            Ok(path.to_string_lossy().to_string())
        }
        None => {
            anyhow::bail!(
                "No model specified. Use --model or provide a subcommand (file/hf-model)"
            );
        }
    }
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Get model path
    let model_path = get_model_path(&args)?;

    // Initialize llama backend
    let _backend = LlamaBackend::init()?;

    // Create server context
    let ctx_server = ServerContext::new().context("Failed to create server context")?;

    // Prepare model params
    let model_params = ServerModelParams {
        n_ctx: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        n_threads: args.threads,
        flash_attn_type: if args.flash_attn {
            llama_cpp_2::server::FlashAttnType::Enabled
        } else {
            llama_cpp_2::server::FlashAttnType::Auto
        },
        system_prompt: args.system_prompt.clone(),
        ..Default::default()
    };

    // Load model
    println!("\nLoading model...");
    ctx_server
        .load_model(&model_path, model_params)
        .context("Failed to load model")?;

    // Get model info
    let meta = ctx_server.get_meta().context("Failed to get model meta")?;

    // Start server loop in background thread
    let ctx_for_loop = ctx_server.clone();
    let server_thread = thread::spawn(move || {
        ctx_for_loop.start_loop();
    });

    // Remember if we have a custom system prompt before moving it
    let has_system_prompt = args.system_prompt.is_some();

    // Create CLI context
    let mut cli = CliContext::new(
        ctx_server.clone(),
        args.system_prompt,
        args.n_predict,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
    );

    // Setup Ctrl+C handler
    let is_interrupted = cli.is_interrupted.clone();
    let ctx_for_signal = ctx_server.clone();
    ctrlc::set_handler(move || {
        if is_interrupted.load(Ordering::Relaxed) {
            // Second Ctrl+C - terminate
            println!("\nForce terminating...");
            ctx_for_signal.terminate();
            std::process::exit(130);
        }
        is_interrupted.store(true, Ordering::Relaxed);
    })?;

    // Print banner
    println!("{}", LLAMA_ASCII_LOGO);
    println!("build      : {}", meta.build_info);
    println!("model      : {}", meta.model_name);
    let mut modalities = String::from("text");
    if meta.has_inp_image {
        modalities.push_str(", vision");
    }
    if meta.has_inp_audio {
        modalities.push_str(", audio");
    }
    println!("modalities : {}", modalities);
    if has_system_prompt {
        println!("using custom system prompt");
    }
    println!();
    println!("available commands:");
    println!("  /exit or Ctrl+C     stop or exit");
    println!("  /regen              regenerate the last response");
    println!("  /clear              clear the chat history");
    println!();

    // Interactive loop
    let stdin = io::stdin();
    let mut current_msg = String::new();

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            // EOF
            break;
        }

        // Remove trailing newline
        let buffer = line.trim_end().to_string();

        // Skip empty messages
        if buffer.is_empty() {
            continue;
        }

        // Check for interrupt
        if cli.is_interrupted.load(Ordering::Relaxed) {
            cli.is_interrupted.store(false, Ordering::Relaxed);
            break;
        }

        let mut add_user_msg = true;

        // Process commands
        if buffer.starts_with("/exit") {
            break;
        } else if buffer.starts_with("/regen") {
            if cli.messages.len() >= 2 {
                // Remove last assistant message
                cli.messages.pop();
                add_user_msg = false;
            } else {
                eprintln!("No message to regenerate.");
                continue;
            }
        } else if buffer.starts_with("/clear") {
            cli.messages.clear();
            current_msg.clear();
            println!("Chat history cleared.");
            continue;
        } else {
            current_msg.push_str(&buffer);
        }

        println!();

        // Generate response
        if add_user_msg {
            cli.messages.push(ChatMessage::new("user", &current_msg));
            current_msg.clear();
        }

        match cli.generate_completion() {
            Ok((assistant_content, timings)) => {
                cli.messages
                    .push(ChatMessage::new("assistant", &assistant_content));
                println!();

                if args.show_timings {
                    println!();
                    println!(
                        "[ Prompt: {:.1} t/s | Generation: {:.1} t/s ]",
                        timings.prompt_per_second, timings.predicted_per_second
                    );
                }
            }
            Err(e) => {
                eprintln!("Error generating response: {}", e);
            }
        }
    }

    println!("\nExiting...");
    ctx_server.terminate();
    server_thread.join().expect("Server thread panicked");

    Ok(())
}
