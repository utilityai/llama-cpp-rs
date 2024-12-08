# Rust Reranker Implementation

A Rust implementation of cross-encoder based reranking using llama-cpp-2. Cross-encoder reranking is a more accurate way to determine similarity between queries and documents compared to traditional embedding-based approaches.

## Overview

This implementation adds a new pooling type `LLAMA_POOLING_TYPE_RANK` which enables cross-encoder based reranking. Unlike traditional embedding approaches that encode query and document separately, this method:

- Processes query and document pairs together in a single pass
- Directly evaluates semantic relationships between the pairs
- Outputs raw similarity scores indicating relevance

## Installation

```bash
# Clone the repository
cd examples/reranker

# Build the project
cargo build --release
```

## Usage

### Command Line Interface

```bash
cargo run --release -- \
    --model-path /path/to/model.gguf \
    --query "what is panda?" \
    --documents "The giant panda is a bear species endemic to China." \
    --pooling rank
```

### CLI Arguments

- `--model-path`: Path to the GGUF model file
- `--query`: The search query
- `--documents`: One or more documents to rank against the query
- `--pooling`: Pooling type (options: none, mean, rank)

### Pooling Types

- `rank`: Performs cross-encoder reranking 

## Example Output

```bash
$ cargo run --release -- \
    --model-path "models/bge-reranker.gguf" \
    --query "what is panda?" \
    --documents "The giant panda is a bear species endemic to China." \
    --pooling rank

rerank score 0: 8.234
```

Note: The raw scores are not normalized through a sigmoid function. If you need scores between 0-1, you'll need to implement sigmoid normalization in your application code.

# Additional notes

- Query and documents are concatenated using the format <bos>query</eos><sep>answer</eos> 

## Supported Models

Some tested models:

- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [jinaai/jina-reranker-v1-tiny-en](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en)

Not tested others, but anything supported by llama.cpp should work. 

## Implementation Details

This is a close Rust implementation of the reranker implementation discussed in [llama.cpp PR #9510](https://github.com/ggerganov/llama.cpp/pull/9510). Key features include:
