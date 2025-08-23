//! Example demonstrating RPC backend support for distributed inference.
//!
//! This example shows how to:
//! - Set up RPC clients to connect to remote servers
//! - Use RPC devices for distributed model execution
//! - Run inference across multiple machines
//!
//! To run this example:
//! 1. Start RPC servers on remote machines using llama.cpp's rpc-server
//! 2. Run this client with the server endpoints

use anyhow::Result;
use clap::{Parser, Subcommand};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    rpc::{RpcBackend, RpcDevice},
    sampling::LlamaSampler,
};
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;

/// Command line arguments for the RPC example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run as an RPC client connecting to remote servers
    Client {
        /// Path to the model file
        #[arg(short = 'm', long = "model")]
        model_path: PathBuf,

        /// RPC server endpoints (can be specified multiple times)
        #[arg(short = 'e', long = "endpoint", required = true, num_args = 1..)]
        endpoints: Vec<String>,

        /// Prompt to use for generation
        #[arg(short = 'p', long = "prompt", default_value = "The meaning of life is")]
        prompt: String,

        /// Number of tokens to generate
        #[arg(short = 'n', long = "n-predict", default_value_t = 128)]
        n_predict: i32,

        /// Context size
        #[arg(short = 'c', long = "ctx-size", default_value_t = 2048)]
        ctx_size: u32,

        /// Temperature for sampling
        #[arg(long = "temp", default_value_t = 0.8)]
        temperature: f32,

        /// Query memory of remote devices before starting
        #[arg(long = "query-memory")]
        query_memory: bool,
    },
    /// Display information about RPC devices
    Info {
        /// RPC server endpoints to query
        #[arg(short = 'e', long = "endpoint", required = true, num_args = 1..)]
        endpoints: Vec<String>,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Client {
            model_path,
            endpoints,
            prompt,
            n_predict,
            ctx_size,
            temperature,
            query_memory,
        } => run_client(
            model_path,
            endpoints,
            prompt,
            n_predict,
            ctx_size,
            temperature,
            query_memory,
        ),
        Commands::Info { endpoints } => show_info(endpoints),
    }
}

fn run_client(
    model_path: PathBuf,
    endpoints: Vec<String>,
    prompt: String,
    n_predict: i32,
    ctx_size: u32,
    temperature: f32,
    query_memory: bool,
) -> Result<()> {
    println!("Initializing RPC client...");

    // Initialize the backend
    let backend = LlamaBackend::init()?;

    // Add RPC devices
    let mut devices = Vec::new();
    for endpoint in &endpoints {
        println!("Adding RPC device: {}", endpoint);
        let device = RpcDevice::add(endpoint)?;
        println!("  Device name: {}", device.name());
        println!("  Description: {}", device.description());
        devices.push(device);
    }

    // Optionally query memory of remote devices
    if query_memory {
        println!("\nQuerying remote device memory:");
        for endpoint in &endpoints {
            // Create a temporary backend to query memory
            if let Ok(rpc_backend) = RpcBackend::init(endpoint) {
                if let Ok((free, total)) = rpc_backend.get_device_memory() {
                    let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
                    let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
                    println!(
                        "  {}: {:.2} GB / {:.2} GB free",
                        endpoint, free_gb, total_gb
                    );
                } else {
                    println!("  {}: Unable to query memory", endpoint);
                }
            }
        }
        println!();
    }

    // Check if model exists
    if !model_path.exists() {
        eprintln!("Error: Model file not found: {}", model_path.display());
        std::process::exit(1);
    }

    println!("Loading model: {}", model_path.display());

    // Set up model parameters
    // When using RPC, the model can be distributed across devices
    let model_params = LlamaModelParams::default()
        .with_split_mode(llama_cpp_2::model::params::LlamaSplitMode::Row);

    // Load the model
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    println!("Model loaded successfully!");

    // Get model info
    let n_vocab = model.n_vocab();
    println!("Model vocabulary size: {}", n_vocab);

    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
        .with_n_threads(4);

    let mut ctx = model.new_context(&backend, ctx_params)?;
    println!("Context created with size: {}", ctx_size);

    // Tokenize the prompt
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    println!("Prompt tokenized into {} tokens", tokens.len());

    // Create batch
    let mut batch = LlamaBatch::new(512, 1);

    // Add tokens to batch
    let last_index = tokens.len() - 1;
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == last_index;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    // Decode the batch
    println!("Processing prompt across {} RPC devices...", endpoints.len());
    ctx.decode(&mut batch)?;

    // Set up sampling
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::top_p(0.95, 1),
    ]);

    // Generate text
    print!("{}", prompt);
    io::stdout().flush()?;

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    while n_decode < n_predict {
        // Sample the next token
        let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(new_token);

        // Check for EOS
        if model.is_eog_token(new_token) {
            println!();
            break;
        }

        // Print the token
        let piece = model.token_to_str(new_token, llama_cpp_2::model::Special::Tokenize)?;
        print!("{}", piece);
        io::stdout().flush()?;

        // Prepare the next batch
        batch.clear();
        batch.add(new_token, n_cur, &[0], true)?;
        n_cur += 1;

        // Decode
        ctx.decode(&mut batch)?;
        n_decode += 1;
    }

    println!("\n\nGeneration complete!");
    println!("Generated {} tokens using {} RPC devices", n_decode, endpoints.len());

    Ok(())
}

fn show_info(endpoints: Vec<String>) -> Result<()> {
    println!("RPC Device Information");
    println!("======================\n");

    for endpoint in endpoints {
        println!("Endpoint: {}", endpoint);
        
        // Try to add as a device
        match RpcDevice::add(&endpoint) {
            Ok(device) => {
                println!("  Status: Connected");
                println!("  Name: {}", device.name());
                println!("  Description: {}", device.description());
            }
            Err(e) => {
                println!("  Status: Failed to connect");
                println!("  Error: {}", e);
            }
        }

        // Try to get memory info
        if let Ok(backend) = RpcBackend::init(&endpoint) {
            if let Ok((free, total)) = backend.get_device_memory() {
                let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
                let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
                let used_gb = (total - free) as f64 / (1024.0 * 1024.0 * 1024.0);
                let usage_percent = (used_gb / total_gb) * 100.0;
                
                println!("  Memory:");
                println!("    Total: {:.2} GB", total_gb);
                println!("    Used:  {:.2} GB ({:.1}%)", used_gb, usage_percent);
                println!("    Free:  {:.2} GB", free_gb);
            } else {
                println!("  Memory: Unable to query");
            }
        }
        
        println!();
    }

    Ok(())
}