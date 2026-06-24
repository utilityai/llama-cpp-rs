# speculative-mtp

MTP (Multi-Token Prediction / NextN) speculative decoding example + benchmark.

Uses a **single** GGUF whose embedded NextN head drafts tokens from the target model's own hidden
state — no separate draft model. The speculative loop lives in Rust (`MtpSpeculator`); the MTP head
forward runs inside llama.cpp.

`--baseline` runs the same model with plain greedy decoding (no speculation), so the two tok/s
numbers are directly comparable. The final line reports `accept%` (fraction of drafted tokens the
target accepted) — `accept% > 0` means the MTP head is drafting useful tokens.

## Recurrent / hybrid architectures (qwen35 etc.)

`qwen35` is a hybrid Gated-DeltaNet + MoE arch. Its recurrent layers carry an O(1) running state
that cannot be partially deleted, so rolling back rejected draft tokens needs bounded recurrent-state
snapshots. The example sets `n_rs_seq = n_draft + 4` on both contexts (via
`LlamaContextParams::with_n_rs_seq`) so partial rollback is a cheap snapshot-index switch. Without
`n_rs_seq > 0`, `seq_rm` silently no-ops on these archs and the loop desyncs after the first step.

## Benchmarks

128 tokens, prompt *"Explain how speculative decoding works in large language models."*, greedy,
`n_draft=4`, model `Qwen3.5-4B-UD-Q4_K_XL.gguf`:

| backend                 | baseline   | MTP        | speedup | accept% |
|-------------------------|------------|------------|---------|---------|
| CPU (32-core)           | 8.9 tok/s  | 11.5 tok/s | 1.29x   | 45.7%   |
| CUDA (RTX 4080)         | 113.0 tok/s| 175.9 tok/s| 1.56x   | 42.7%   |
| Metal (Apple Silicon)   | 22.4 tok/s | 33.8 tok/s | 1.51x   | 45.7%   |

## Run

CPU:

```bash
cargo run -p speculative-mtp --release -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --n-predict 128 --n-gpu-layers 0
# add --baseline for the no-speculation baseline
```

CUDA:

```bash
cargo run -p speculative-mtp --release --features cuda -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf --n-predict 128 --n-gpu-layers 999
```

Metal:

```bash
cargo run -p speculative-mtp --release --features metal -- \
  --model /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf --n-predict 128 --n-gpu-layers 999
```

Vulkan: same as above with `--features vulkan`.

## Flags

- `--n-draft N` — max draft tokens per step (default 4).
- `--n-rs-seq N` — recurrent-state rollback snapshots (default `0` = auto = `n_draft + 4`).
- `--baseline` — plain greedy decode, no speculation.
- `--quiet` — suppress generated text, print only the benchmark line.
