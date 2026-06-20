# speculative-mtp

MTP (Multi-Token Prediction / NextN) speculative decoding smoke test.

Uses a **single** GGUF whose embedded NextN head drafts tokens from the target model's own hidden
state — no separate draft model. The speculative loop lives in Rust (`MtpSpeculator`); the MTP head
forward runs inside llama.cpp.

The final line reports `accept%`, the fraction of drafted tokens the target accepted. `accept% > 0`
means the MTP head is actually drafting useful tokens.

## Per-backend probes

CPU (correctness baseline):

```bash
cargo run -p speculative-mtp --release -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --n-predict 48 --n-gpu-layers 0
```

CUDA:

```bash
cargo run -p speculative-mtp --release --features cuda -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "Explain speculative decoding in one paragraph." \
  --n-predict 96 --n-gpu-layers 999
```

Metal (Apple Silicon):

```bash
cargo run -p speculative-mtp --release --features metal -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --n-predict 48 --n-gpu-layers 999
```

Vulkan:

```bash
cargo run -p speculative-mtp --release --features vulkan -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --n-predict 48 --n-gpu-layers 999
```

## Expected output

A coherent continuation of the prompt, then:

```
[mtp] generated=48 n_drafted=... n_accept=... accept=NN.N% time=...s speed=... tok/s
```

If a backend lacks a kernel for the `qwen35` hybrid (Gated DeltaNet) arch, ggml logs a CPU fallback
(slower, still correct). A hard failure to create the context or decode indicates a missing/broken
backend op for this arch — separate from MTP itself.
