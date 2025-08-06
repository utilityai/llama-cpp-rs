//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};

/// Supported embedding model architectures
#[derive(Debug, Clone, Copy, PartialEq)]
enum EmbeddingArchitecture {
    /// BAAI General Embedding - uses CLS token pooling
    BGE,
    /// General Text Embeddings - uses mean pooling
    GTE,
    /// Qwen3 Embedding - uses last token pooling
    Qwen3,
    /// JINA Embeddings - uses mean pooling
    JINA,
    /// Nomic Embed - uses mean pooling with task prefixes
    Nomic,
    /// Unknown architecture - will use default pooling based on model metadata
    Unknown,
}

impl EmbeddingArchitecture {
    /// Detect architecture from model path
    fn detect(model_path: &Path) -> Self {
        let path_str = model_path.to_string_lossy().to_lowercase();
        
        if path_str.contains("bge") {
            Self::BGE
        } else if path_str.contains("gte") {
            Self::GTE
        } else if path_str.contains("qwen3") {
            Self::Qwen3
        } else if path_str.contains("jina") {
            Self::JINA
        } else if path_str.contains("nomic") {
            Self::Nomic
        } else {
            Self::Unknown
        }
    }
    
    /// Get the expected pooling type for this architecture
    fn pooling_type(&self) -> llama_cpp_2::context::params::LlamaPoolingType {
        use llama_cpp_2::context::params::LlamaPoolingType;
        
        match self {
            Self::BGE => LlamaPoolingType::Cls,
            Self::GTE => LlamaPoolingType::Mean,
            Self::Qwen3 => LlamaPoolingType::Last,
            Self::JINA => LlamaPoolingType::Mean,
            Self::Nomic => LlamaPoolingType::Mean,
            Self::Unknown => LlamaPoolingType::None, // Let model decide
        }
    }
    
    /// Whether this architecture requires L2 normalization
    fn requires_normalization(&self) -> bool {
        match self {
            Self::BGE | Self::GTE | Self::Qwen3 | Self::JINA | Self::Nomic => true,
            Self::Unknown => false,
        }
    }
}

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The prompt
    #[clap(default_value = "Hello my name is")]
    prompt: String,
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// Read prompts from stdin (one per line)
    #[clap(long)]
    stdin: bool,
    /// Whether to normalise the produced embeddings (overrides architecture default)
    #[clap(short)]
    normalise: bool,
    /// Force normalization off (useful for architectures that default to normalizing)
    #[clap(long)]
    no_normalise: bool,
    /// Output embeddings as JSON
    #[clap(long)]
    json: bool,
    /// Task type for Nomic models (query or document)
    #[clap(long, default_value = "document")]
    task_type: String,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `BAAI/bge-small-en-v1.5`
        repo: String,
        /// the model name. e.g. `BAAI-bge-small-v1.5.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

fn main() -> Result<()> {
    let Args {
        prompt,
        model,
        stdin,
        normalise,
        no_normalise,
        json,
        task_type,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
    } = Args::parse();

    // init LLM
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();  // Suppress llama.cpp logs for cleaner output

    // offload all layers to the gpu
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let model_path = model
        .get_or_load()
        .with_context(|| "failed to get model from args")?;

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // Detect architecture
    let architecture = EmbeddingArchitecture::detect(&model_path);
    eprintln!("Detected architecture: {:?}", architecture);

    // initialize the context
    let mut ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);

    // Set pooling type based on architecture
    let expected_pooling = architecture.pooling_type();
    if !matches!(expected_pooling, llama_cpp_2::context::params::LlamaPoolingType::None) {
        eprintln!("Setting pooling type to {:?} for {:?}", expected_pooling, architecture);
        ctx_params = ctx_params.with_pooling_type(expected_pooling);
    }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;
    
    // Determine if we should normalize based on architecture and flags
    let should_normalize = if no_normalise {
        false
    } else if normalise {
        true
    } else {
        architecture.requires_normalization()
    };
    eprintln!("Normalization: {}", if should_normalize { "enabled" } else { "disabled" });

    // Get prompts either from stdin or command line
    let prompts: Vec<String> = if stdin {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        stdin.lock().lines().collect::<Result<Vec<_>, _>>()?
    } else {
        prompt.lines().map(|s| s.to_string()).collect()
    };

    // tokenize the prompts
    let tokens_lines_list = prompts
        .iter()
        .map(|line| {
            // Add task prefix for Nomic models
            let text_to_encode = if matches!(architecture, EmbeddingArchitecture::Nomic) {
                let prefix = if task_type == "query" {
                    "search_query: "
                } else {
                    "search_document: "
                };
                format!("{}{}", prefix, line)
            } else {
                line.to_string()
            };
            
            model.str_to_token(&text_to_encode, AddBos::Always)
        })
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| "failed to tokenize prompts")?;

    let n_ctx = ctx.n_ctx() as usize;
    let n_ctx_train = model.n_ctx_train();

    eprintln!("n_ctx = {n_ctx}, n_ctx_train = {n_ctx_train}");

    if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
        bail!("One of the provided prompts exceeds the size of the context window");
    }

    // print the prompt token-by-token
    eprintln!();

    for (i, token_line) in tokens_lines_list.iter().enumerate() {
        eprintln!("Prompt {i}");
        for token in token_line {
            // Attempt to convert token to string and print it; if it fails, print the token instead
            match model.token_to_str(*token, Special::Tokenize) {
                Ok(token_str) => eprintln!("{token} --> {token_str}"),
                Err(e) => {
                    eprintln!("Failed to convert token to string, error: {e}");
                    eprintln!("Token value: {token}");
                }
            }
        }
        eprintln!();
    }

    std::io::stderr().flush()?;

    // create a llama_batch with the size of the context
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(n_ctx, 1);

    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    let t_main_start = ggml_time_us();

    for tokens in &tokens_lines_list {
        // Flush the batch if the next prompt would exceed our batch size
        if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
            batch_decode(
                &mut ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                should_normalize,
            )?;
            max_seq_id_batch = 0;
        }

        batch.add_sequence(tokens, max_seq_id_batch, true)?;
        max_seq_id_batch += 1;
    }
    // Handle final batch
    batch_decode(
        &mut ctx,
        &mut batch,
        max_seq_id_batch,
        &mut output,
        should_normalize,
    )?;

    let t_main_end = ggml_time_us();

    if json {
        // Output as JSON array
        println!("[");
        for (i, embeddings) in output.iter().enumerate() {
            print!("  [");
            for (j, val) in embeddings.iter().enumerate() {
                if j > 0 { print!(", "); }
                print!("{}", val);
            }
            print!("]");
            if i < output.len() - 1 { println!(","); } else { println!(); }
        }
        println!("]");
    } else {
        for (i, embeddings) in output.iter().enumerate() {
            eprintln!("Embeddings {i}: {:?}", embeddings);
            eprintln!("First 10 values: {:?}", &embeddings[..embeddings.len().min(10)]);
            
            // Calculate L2 norm
            let norm = embeddings.iter()
                .fold(0.0, |acc, &val| acc + val * val)
                .sqrt();
            eprintln!("L2 norm: {}", norm);
            eprintln!();
        }
    }

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    eprintln!(
        "Created embeddings for {} tokens in {:.2} s, speed {:.2} t/s\n",
        total_tokens,
        duration.as_secs_f32(),
        total_tokens as f32 / duration.as_secs_f32()
    );

    if !json {
        println!("{}", ctx.timings());
    }

    Ok(())
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    use llama_cpp_2::context::params::LlamaPoolingType;
    
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    let pooling_type = ctx.pooling_type();
    eprintln!("Pooling type: {:?}", pooling_type);

    for i in 0..s_batch {
        let embedding = match pooling_type {
            LlamaPoolingType::None => {
                // For models with no pooling, use token embeddings
                ctx.embeddings_ith(i)
                    .with_context(|| "Failed to get token embeddings")?
            }
            _ => {
                // For models with pooling (Mean, CLS, Last, etc.), use sequence embeddings
                ctx.embeddings_seq_ith(i)
                    .with_context(|| "Failed to get sequence embeddings")?
            }
        };
        
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

fn normalize(input: &[f32]) -> Vec<f32> {
    // Calculate L2 norm (magnitude)
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    // Avoid division by zero
    if magnitude == 0.0 {
        return input.to_vec();
    }

    // L2 normalization
    input.iter().map(|&val| val / magnitude).collect()
}
