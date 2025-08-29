//! Example demonstrating how to load split GGUF models.
//!
//! This example shows how to:
//! - Load a model split across multiple files
//! - Use utility functions to work with split file naming conventions
//! - Generate text from a split model

use anyhow::Result;
use clap::Parser;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;

/// Command line arguments for the split model example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Paths to the split model files (can be specified multiple times)
    #[arg(short = 'm', long = "model", required = true, num_args = 1..)]
    model_paths: Vec<PathBuf>,

    /// Alternatively, provide a prefix and the program will auto-detect splits
    #[arg(short = 'p', long = "prefix", conflicts_with = "model_paths")]
    prefix: Option<String>,

    /// Number of splits (required if using --prefix)
    #[arg(short = 'n', long = "num-splits", requires = "prefix")]
    num_splits: Option<u32>,

    /// Prompt to use for generation
    #[arg(short = 't', long = "prompt", default_value = "Once upon a time")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 'g', long = "n-predict", default_value_t = 128)]
    n_predict: i32,

    /// Number of GPU layers
    #[arg(short = 'l', long = "n-gpu-layers", default_value_t = 0)]
    n_gpu_layers: u32,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value_t = 2048)]
    ctx_size: u32,

    /// Temperature for sampling
    #[arg(long = "temp", default_value_t = 0.8)]
    temperature: f32,

    /// Top-P for sampling
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Seed for random number generation
    #[arg(long = "seed", default_value_t = 1234)]
    seed: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Determine the model paths
    let model_paths = if let Some(prefix) = args.prefix {
        let num_splits = args.num_splits.expect("num-splits required with prefix");
        
        // Generate split paths using the utility function
        let mut paths = Vec::new();
        for i in 1..=num_splits {
            let path = LlamaModel::split_path(&prefix, i as i32, num_splits as i32);
            paths.push(PathBuf::from(path));
        }
        
        println!("Generated split paths:");
        for path in &paths {
            println!("  - {}", path.display());
        }
        
        paths
    } else {
        args.model_paths
    };

    // Verify all split files exist
    for path in &model_paths {
        if !path.exists() {
            eprintln!("Error: Split file not found: {}", path.display());
            std::process::exit(1);
        }
    }

    println!("Loading model from {} splits...", model_paths.len());

    // Initialize the backend
    let backend = LlamaBackend::init()?;

    // Set up model parameters
    let mut model_params = LlamaModelParams::default();
    if args.n_gpu_layers > 0 {
        model_params = model_params.with_n_gpu_layers(args.n_gpu_layers);
    }

    // Load the model from splits
    let model = LlamaModel::load_from_splits(&backend, &model_paths, &model_params)?;
    println!("Model loaded successfully!");

    // Get model info
    let n_vocab = model.n_vocab();
    println!("Model vocabulary size: {}", n_vocab);

    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(args.ctx_size).unwrap()));

    let mut ctx = model.new_context(&backend, ctx_params)?;
    println!("Context created with size: {}", args.ctx_size);

    // Tokenize the prompt
    let tokens = model.str_to_token(&args.prompt, AddBos::Always)?;
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
    ctx.decode(&mut batch)?;
    println!("Initial prompt processed");

    // Set up sampling
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(args.temperature),
        LlamaSampler::top_p(args.top_p, 1),
    ]);

    // Generate text
    print!("{}", args.prompt);
    io::stdout().flush()?;

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    while n_decode < args.n_predict {
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
    println!("Generated {} tokens", n_decode);

    // Demonstrate the split_prefix utility
    if let Some(first_path) = model_paths.first() {
        if let Some(path_str) = first_path.to_str() {
            // Try to extract the prefix from the first split file
            if let Some(prefix) = LlamaModel::split_prefix(path_str, 1, model_paths.len() as i32) {
                println!("\nExtracted prefix from first split: {}", prefix);
            }
        }
    }

    Ok(())
}