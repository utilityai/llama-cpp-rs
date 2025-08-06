# Embedding Architecture Guide

This guide explains how the llama-cpp-rs embeddings example handles different embedding model architectures.

## 🔬 Benchmarks

For comprehensive performance and accuracy comparisons, see the [benchmarks](benchmarks/) directory. The benchmarks compare this Rust implementation against Python (sentence-transformers) and ONNX implementations with proper methodology.

## Supported Architectures

The following embedding architectures are supported with automatic detection:

| Architecture | Pooling Method | Normalization | Special Features |
|-------------|----------------|---------------|-----------------|
| **BGE** | CLS Token | Yes (L2) | Uses [CLS] token at position 0 |
| **GTE** | Mean | Optional | Average of all token embeddings |
| **Qwen3** | Last Token | Yes (L2) | Uses EOS token position |
| **JINA** | Mean | Yes (L2) | Advanced mean pooling |
| **Nomic** | Mean | Yes (L2) | Requires task prefixes |

## Architecture Detection

The system automatically detects the architecture based on the model file name:

```rust
let architecture = EmbeddingArchitecture::detect(&model_path);
```

Detection is case-insensitive and looks for architecture names in the file path:
- Files containing "bge" → BGE architecture
- Files containing "gte" → GTE architecture
- Files containing "qwen3" → Qwen3 architecture
- Files containing "jina" → JINA architecture
- Files containing "nomic" → Nomic architecture

## Pooling Methods

### CLS Token Pooling (BGE)
Takes the embedding from the first token position ([CLS] token):
```rust
ctx.embeddings_seq_ith(0)  // For first sequence
```

### Mean Pooling (GTE, JINA, Nomic)
Averages embeddings across all non-padding tokens:
```rust
ctx.embeddings_seq_ith(i)  // llama.cpp handles mean pooling internally
```

### Last Token Pooling (Qwen3)
Takes the embedding from the last non-padding token (EOS position):
```rust
ctx.embeddings_seq_ith(i)  // llama.cpp handles last token selection
```

## Normalization

L2 normalization is applied based on architecture requirements:

```rust
fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input.iter()
        .fold(0.0, |acc, &val| acc + val * val)
        .sqrt();
    
    if magnitude == 0.0 {
        return input.to_vec();
    }
    
    input.iter().map(|&val| val / magnitude).collect()
}
```

Default normalization by architecture:
- **BGE, Qwen3, JINA, Nomic**: L2 normalization enabled
- **GTE**: Normalization disabled (optional)

Override with command-line flags:
- `-n` or `--normalise`: Force normalization on
- `--no-normalise`: Force normalization off

## Special Features

### Task Prefixes (Nomic)

Nomic models require task-specific prefixes:
```rust
// For documents (default)
"search_document: <text>"

// For queries
"search_query: <text>"
```

Use `--task-type query` or `--task-type document` to control the prefix.

## Usage Examples

```bash
# BGE model (CLS pooling, normalized)
cargo run -p embeddings -- "Hello world" local /path/to/bge-model.gguf

# GTE model (mean pooling, no normalization by default)
cargo run -p embeddings -- "Hello world" local /path/to/gte-model.gguf

# Qwen3 model (last token pooling, normalized)
cargo run -p embeddings -- "Hello world" local /path/to/qwen3-model.gguf

# JINA model (mean pooling, normalized)
cargo run -p embeddings -- "Hello world" local /path/to/jina-model.gguf

# Nomic model with query prefix
cargo run -p embeddings -- "Hello world" local /path/to/nomic-model.gguf --task-type query

# Force normalization off for any model
cargo run -p embeddings -- "Hello world" local /path/to/model.gguf --no-normalise
```

## Testing and Verification

The `py/` directory contains Python scripts to verify embeddings match the reference implementations:

```bash
cd examples/embeddings/py

# Run Python reference
python test_bge.py

# Compare with Rust implementation
python test_single.py --model bge --gguf /path/to/bge-model.gguf
```

Expected differences due to quantization:
- Q8_0: < 5% difference
- Q5_K_M: < 10% difference
- Q4_0: < 15% difference

## Adding New Architectures

To add support for a new architecture:

1. Add to the `EmbeddingArchitecture` enum
2. Update the `detect()` method with detection logic
3. Map to appropriate pooling type in `pooling_type()`
4. Set normalization requirement in `requires_normalization()`
5. Add any special handling (like task prefixes)
6. Create a Python test script for verification

## Technical Notes

- Special tokens ([CLS], [SEP], EOS) are handled by llama.cpp's tokenizer
- The context must be created with `.with_embeddings(true)`
- Pooling type can be explicitly set with `.with_pooling_type()`
- llama.cpp may not automatically detect pooling types for all models
- Architecture detection provides a workaround for models with incorrect metadata