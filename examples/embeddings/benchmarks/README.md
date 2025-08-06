# Embedding Benchmarks

This directory contains comprehensive benchmarks for comparing embedding implementations with fair methodology.

## Overview

The benchmark compares three implementations:
- **Python** (sentence-transformers) - Baseline reference with full precision
- **Rust** (llama-cpp-rs) - This crate with quantized GGUF models
- **ONNX** (EmbedAnything) - PyTorch models via ONNX Runtime (same precision as Python)

## Key Features

- **Fair Comparison**: Models loaded once and reused (not reloaded per text)
- **Comprehensive**: Tests 5 major models (BGE, GTE, JINA, QWEN3, NOMIC) with 50 texts
- **Both Metrics**: Performance (timing) AND accuracy (cosine similarity)
- **CSV Export**: Dimension-by-dimension comparison for analysis
- **Hardware Acceleration**: Metal for GGUF, CoreML for ONNX
- **Graceful Error Handling**: Missing models or dependencies are skipped with clear messages
- **Easy Configuration**: JSON config file for model paths and settings

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install sentence-transformers embed-anything numpy

# Build the embeddings binary (from repo root)
cargo build --release --bin embeddings --features metal  # macOS
cargo build --release --bin embeddings --features cuda   # NVIDIA GPU
cargo build --release --bin embeddings                   # CPU only
```

### Configure Model Paths

Copy `config-example.json` to `config.json` and update with paths to your GGUF models:

```bash
cp config-example.json config.json
# Edit config.json with your model paths
```

The configuration file structure:

```json
{
  "models": {
    "BGE": {
      "gguf": "~/models/bge-small-en-v1.5-q8_0.gguf",
      "hf": "BAAI/bge-small-en-v1.5",
      "normalize": true,
      "cli_pooling": "cls"
    }
  },
  "benchmark_settings": {
    "max_csv_texts": 5,
    "max_csv_dimensions": 50,
    "llama_threads": 6,
    "enable_metal": true,
    "ort_execution_providers": "CoreMLExecutionProvider,CPUExecutionProvider"
  }
}
```

Download models from Hugging Face (Q8_0 or Q4_K_M quantization recommended):
- [BGE GGUF](https://huggingface.co/BAAI/bge-small-en-v1.5-GGUF)
- [GTE GGUF](https://huggingface.co/thenlper/gte-base-GGUF) 
- [JINA GGUF](https://huggingface.co/jinaai/jina-embeddings-v2-base-en-GGUF)
- [QWEN3 GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [NOMIC GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)

### Run Benchmark

```bash
cd examples/embeddings/benchmarks
python comprehensive_benchmark.py
```

## Configuration

The benchmark uses a `config.json` file for easy configuration management:

### Model Configuration
Each model in the config has the following fields:
- `gguf`: Path to the GGUF model file (supports `~` for home directory)
- `hf`: Hugging Face model identifier for Python/ONNX
- `normalize`: Whether to apply L2 normalization (true/false)
- `cli_pooling`: Pooling method for CLI ("cls", "mean", "last")
- `trust_remote_code`: Whether to trust remote code (for models that require it)

### Benchmark Settings
- `max_csv_texts`: Number of texts to export in CSV (default: 5)
- `max_csv_dimensions`: Number of embedding dimensions to export (default: 50)
- `llama_threads`: Number of threads for llama.cpp (default: 6)
- `enable_metal`: Enable Metal acceleration on macOS (true/false)
- `ort_execution_providers`: ONNX Runtime execution providers

### Platform-Specific Settings

**macOS**: Use Metal for GGUF and CoreML for ONNX:
```json
"enable_metal": true,
"ort_execution_providers": "CoreMLExecutionProvider,CPUExecutionProvider"
```

**Linux/Windows with NVIDIA**: Use CUDA for GGUF and GPU for ONNX:
```json
"enable_metal": false,
"ort_execution_providers": "CUDAExecutionProvider,CPUExecutionProvider"
```

**CPU-only**: Disable hardware acceleration:
```json
"enable_metal": false,
"ort_execution_providers": "CPUExecutionProvider"
```

## Expected Results

### Performance (Average ms per text)
When models are properly loaded once and reused:

| Model | Python | Rust  | ONNX  |
|-------|--------|-------|-------|
| BGE   | 30-40  | 15-25 | 15-20 |
| GTE   | 25-35  | 20-30 | 20-25 |
| JINA  | 30-45  | 25-35 | 25-30 |
| QWEN3 | 120-140| 25-35 | 30-40 |
| NOMIC | 35-45  | 20-30 | 25-30 |

### Accuracy (Cosine Similarity vs Python)
| Implementation | Similarity | Notes |
|----------------|------------|-------|
| Rust (Q8_0)    | >0.995     | Quantization effects |
| Rust (Q6_K)    | >0.85      | More quantization effects |
| ONNX           | 1.000      | Same precision as Python (4/5 models compatible) |

## Key Findings

1. **Rust (GGUF) Performance**: 2-8x faster than Python for inference
2. **ONNX Performance**: 2-6x faster than Python (but same precision - likely cached PyTorch)
3. **Model Loading**: GGUF loads 10-30x faster than Python
4. **Accuracy**: Q8_0 quantization maintains >99.5% similarity, Q6_K maintains >85% similarity
5. **ONNX Compatibility**: Works with BGE, GTE, JINA, QWEN3 (NOMIC has config incompatibility)
6. **Fair Comparison Critical**: Previous benchmarks were unfair (reloaded models per text)

## Output Files

- **Console Output**: Performance summary and accuracy metrics
- **CSV Files**: `embeddings_comparison_[MODEL].csv` with dimension-by-dimension comparisons
  - Compare Python_Dim000 vs Rust_Dim000 vs ONNX_Dim000 for same text
  - Analyze quantization effects on individual embedding values

## Implementation Notes

### Rust Implementation
- Uses the embeddings binary built from this crate
- Proper batch processing with `--stdin` and `--json` flags  
- Hardware acceleration via Metal/CUDA features
- Single model load with JSON array output

### ONNX Implementation
- Uses EmbedAnything library for model loading
- CoreML provider for hardware acceleration on macOS
- Single model instance reused for all texts
- File-based API (limitation of EmbedAnything)

### Python Implementation
- Uses sentence-transformers (reference baseline)
- Proper batching with model.encode()
- Hardware acceleration via PyTorch/Transformers
- Standard approach for comparison

## Fair Benchmarking Methodology

### ✅ Correct (This Benchmark)
1. Load model once
2. Process all texts with same model instance
3. Measure only inference time
4. Compare apples-to-apples

### ❌ Incorrect (Common Mistake)
1. Spawn new process per text
2. Reload model for each text
3. Include model loading in timing
4. Unfair advantage to Python

## Use Case Recommendations

### Use Rust (GGUF) When:
- Fast startup critical (100ms vs 5000ms model loading)
- Memory constrained (5x less usage due to quantization)
- Edge/mobile deployment
- Single binary deployment preferred

### Use ONNX When:
- Cross-platform compatibility required
- Integration with ONNX ecosystem
- Hardware acceleration available

### Use Python When:
- Development/research phase
- Maximum model ecosystem access
- Existing PyTorch/Transformers workflow

## Contributing

When adding new models or implementations:
1. Ensure fair comparison (model loaded once)
2. Test with multiple models and texts
3. Include both performance and accuracy metrics
4. Update documentation with findings

## Troubleshooting

**Config file not found**: Ensure `config.json` exists in the benchmarks directory

**Invalid JSON**: Check `config.json` syntax with a JSON validator

**Binary not found**: Build with `cargo build --release --bin embeddings`

**Model not found**: Update paths in `config.json` - check that files exist at specified paths

**Import errors**: Install missing dependencies with pip

**Poor performance**: Ensure hardware acceleration is enabled in `config.json` and build with appropriate features (Metal/CUDA)

**Models skipped**: The benchmark gracefully handles missing models or failed loads - check output for specific error messages and update `config.json` as needed