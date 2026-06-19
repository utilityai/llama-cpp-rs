# MTP Speculative Decoding — Design

Date: 2026-06-19
Branch: `feature/speculative-mtp`
Status: approved (proceed to implementation)

## Summary

Expose llama.cpp's already-implemented **MTP (Multi-Token Prediction / NextN) speculative
decoding** through this fork so downstream Rust code can run an MTP speculative loop with a
single self-contained GGUF (no separate draft model). This is **binding work only** — the C++
graph and draft logic already exist in the pinned vendored tree.

## Key facts established during exploration

- Vendored llama.cpp is pinned at commit `9e3b928` (2026-06-07), shipped with
  `llama-cpp-sys-2` 0.1.151. A prior analysis referenced sys 0.1.146, whose tree predates the
  MTP implementation; that analysis is **stale**.
- MTP is fully implemented upstream in this tree:
  - `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` — `common/common.h:163`
  - `common_speculative_impl_draft_mtp` — `common/speculative.cpp:410`
  - MTP forward graph — `LLM_GRAPH_TYPE_DECODER_MTP`, `LLAMA_CONTEXT_TYPE_MTP`
  - The old "TODO: when MTP is implemented" comment is gone.
- The target model `Qwen3.5-4B-UD-Q4_K_XL.gguf` is a **single GGUF**, arch `qwen35`
  (hybrid Gated DeltaNet + MoE), with `qwen35.nextn_predict_layers` metadata. The MTP head is
  **embedded** — there is no separate `mtp-` sibling file.
- The MTP draft context is a **second context on the same model**, created with
  `ctx_type = LLAMA_CONTEXT_TYPE_MTP` and `ctx_other = ctx_tgt` (shared memory with the target).
  Reference: `tools/server/server-context.cpp:962-981`.
- Public-ABI bindings already exist (bindgen sees `include/llama.h`):
  `LLAMA_CONTEXT_TYPE_MTP`, `llama_context_params.ctx_type`, `.ctx_other`, `.n_rs_seq`,
  `.n_outputs_max`, `llama_model_n_embd_out`, `llama_set_sampler`.
- `common_speculative_*` lives in **libcommon** (not the `llama.h` ABI), so it must be reached
  through a hand-written `extern "C"` shim — the same pattern already used for grammar / fit /
  memory-breakdown in `wrapper_common.cpp`. That file is compiled only under the **default
  `common` feature**; MTP reuses that feature (no new feature flag).
- The canonical MTP host loop is in `tools/server/server-context.cpp`. The
  `examples/speculative-simple` example does **not** wire MTP (`speculative-simple.cpp:232`
  says "see server code for reference").

## Backend support (qwen35 + MTP)

The binding is backend-agnostic: the same `llama_decode` / `common_speculative_*` calls run on
every backend; there is **no per-backend MTP code**. Backend risk is therefore essentially the
risk of the base `qwen35` hybrid arch, not of MTP (whose extra ops — eh_proj matmul, norms,
shared head — are standard and universally supported).

Kernel coverage at commit `9e3b928` for the exotic Gated DeltaNet primitives:

| op       | CUDA | Metal | Vulkan | CPU |
|----------|------|-------|--------|-----|
| SSM_SCAN | yes  | yes   | yes    | yes |
| SSM_CONV | yes  | yes   | yes    | yes |
| CUMSUM   | yes  | yes   | yes    | yes |
| TRI      | yes  | yes   | yes    | yes |
| L2_NORM  | yes  | yes   | yes    | yes |

All present on all three GPU backends. `ggml_backend_sched` also auto-offloads any unsupported
op to CPU (degrade, not crash). Residual risk (numerics/NaN, MTP embd-batch shape paths,
flash-attn interaction) can only be closed empirically per device.

Verification plan: empirical CPU + CUDA smoke run on this machine (RTX 4080); empirical Metal
smoke run on `ssh macbook-local` (Apple Silicon, Xcode + cargo present); a documented
one-command per-device probe in the example README for Vulkan / future devices.

## Architecture — three layers

### 1. sys C shim (`llama-cpp-sys-2/wrapper_common.{cpp,h}`, behind default `common` feature)

Flat `extern "C"` `llama_rs_speculative_*` wrappers over `common_speculative`. C++ vectors and
the reference-returning `common_speculative_get_draft_params` are flattened to C arrays + status
codes at this boundary. Exceptions are caught and converted to `llama_rs_status`.

```c
// opaque handle == common_speculative*
typedef struct llama_rs_speculative llama_rs_speculative;

// builds common_params_speculative{ types={DRAFT_MTP},
//   draft={ctx_tgt, ctx_dft, n_max, n_min, p_min, backend_sampling} } and inits.
llama_rs_speculative * llama_rs_speculative_init_mtp(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t  n_max,
    int32_t  n_min,
    float    p_min,
    bool     backend_sampling);

void            llama_rs_speculative_free(llama_rs_speculative * spec);
bool            llama_rs_speculative_need_embd_nextn(const llama_rs_speculative * spec);

llama_rs_status llama_rs_speculative_begin(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    const llama_token * prompt, size_t prompt_len);

// call after each target llama_decode of the verify batch
llama_rs_status llama_rs_speculative_process(
    llama_rs_speculative * spec, const struct llama_batch * batch);

// sets draft params for seq_id, runs draft(), copies tokens into out_buf
llama_rs_status llama_rs_speculative_draft(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last,
    const llama_token * prompt, size_t prompt_len,
    llama_token * out_buf, size_t out_cap, size_t * out_len);

llama_rs_status llama_rs_speculative_accept(
    llama_rs_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);
```

`init_mtp` calls `common_speculative_n_max()` so callers can size `out_buf`; alternatively a
`llama_rs_speculative_n_max(spec)` accessor is exposed. Bindgen already allowlists `llama_rs_.*`.

### 2. llama-cpp-2 safe layer

- **`context/params.rs` + `context/params/get_set.rs`**: add a `LlamaContextType` enum
  (`Default` / `Mtp`) mapping to `llama_context_type`, plus `with_context_type(LlamaContextType)`
  and `context_type()` accessors. `ctx_other` is a raw `*mut llama_context`; it is **not** a
  loose public setter (it would break the existing `Send`/`Sync` justification) — instead it is
  set internally by the MTP-context constructor below, which ties the borrow.
- **new `speculative.rs` module** (declared in `lib.rs`):
  - `LlamaModel::new_mtp_context<'a>(&'a self, target: &'a LlamaContext, params) -> Result<LlamaContext<'a>>`
    — creates the draft context on the **same model** with `ctx_type = Mtp` and
    `ctx_other = target.context.as_ptr()`, mirroring `server-context.cpp:962-981`. The returned
    context borrows both the model and the target context so it cannot outlive them.
  - `MtpSpeculator<'a>` — owns the `*llama_rs_speculative`, borrows target + draft contexts.
    - `begin(&mut self, seq_id, &[LlamaToken]) -> Result<()>`
    - `process(&mut self, &LlamaBatch) -> Result<()>`
    - `draft(&mut self, seq_id, n_past, id_last, prompt: &[LlamaToken]) -> Result<Vec<LlamaToken>>`
    - `accept(&mut self, seq_id, n_accepted: u16) -> Result<()>`
    - `need_embd_nextn(&self) -> bool`, `n_max(&self) -> i32`
    - `Drop` frees the handle.
  - `SpeculativeError` (`thiserror`): `InvalidArgument`, `DecodeFailed`, `BufferTooSmall`,
    `NullHandle`, `InitFailed`, mapped from `llama_rs_status`.

### 3. Example / smoke test — `examples/speculative-mtp/`

A `clap` binary: load model → build target context + MTP draft context → run the loop → print
generated text and acceptance statistics (`n_drafted`, `n_accept`, `accept%`, tokens/s).
Doubles as the manual smoke test. New workspace member in the root `Cargo.toml`.

## Data flow — the speculative loop (mirrors the server MTP path)

Single sequence, per generation step:

1. **Draft** — `spec.draft(seq, n_past, id_last, &prompt)` → draft tokens (seeded from the
   hidden state captured by the previous `process`; on the first step, from prefill).
2. **Verify batch** — build `[id_last @ n_past, draft0 @ n_past+1, …]`, all `logits = true`,
   `target.decode(batch)`.
3. **Process** — `spec.process(&verify_batch)` captures the target hidden states for the *next*
   draft. (Must run after every target decode.)
4. **Accept (in Rust)** — for each position walk the target logits, pick the target token
   (caller's sampler; greedy in the example), accept the longest prefix that matches the draft;
   the first mismatch's target token is the bonus token. Acceptance lives in Rust — the C++
   `common_sampler_sample_and_accept_n` is intentionally **not** used, so callers keep their own
   sampling/grammar/streaming.
5. **Accept bookkeeping** — `spec.accept(seq, n_accepted)` rolls the draft recurrent state.
   `n_accepted` excludes the bonus token.
6. **Commit** — append accepted tokens to output, advance `n_past`, set `id_last` to the last
   accepted token, trim KV beyond `n_past` on both contexts. Loop until EOG or `n_predict`.

Invariants locked against `tools/server/server-context.cpp` during planning: `process` always
runs after the target decode; `draft` consumes the previously-captured hidden state; `accept`'s
count excludes the bonus token; KV is trimmed past `n_past` on both contexts each step.

## Error handling

- Shim: null-checks, `llama_decode` return codes, `out_buf` capacity check → `llama_rs_status`;
  all C++ exceptions caught at the boundary (never unwind into Rust).
- Rust: every shim call returns `Result<_, SpeculativeError>`; no panics across FFI. The MTP
  context constructor returns `Err` if `llama_init_from_model` returns null or if
  `n_embd_out(dft) != n_embd(tgt)`.

## Testing

- **Unit tests (no model, run in CI)** in `llama-cpp-2`:
  - `LlamaContextType` ↔ `llama_context_type` round-trip.
  - `with_context_type` / `context_type()` set the raw field correctly.
  - `SpeculativeError` mapping from each `llama_rs_status`.
  - Draft-buffer capacity / `BufferTooSmall` logic (testable at the Rust boundary).
- **Smoke test (needs model; `#[ignore]` + env-gated)**:
  - Run `examples/speculative-mtp` on `Qwen3.5-4B-UD-Q4_K_XL.gguf`.
  - Assert: non-empty coherent output; **accept% > 0** (proves the MTP head actually drafts).
  - Backends: CPU (correctness) and CUDA (RTX 4080) on this machine; Metal on
    `ssh macbook-local`. Vulkan documented as a user-run probe.

## Files touched

- `llama-cpp-sys-2/wrapper_common.cpp`, `wrapper_common.h` — add the `llama_rs_speculative_*` shim.
- `llama-cpp-2/src/context/params.rs` — `LlamaContextType` enum.
- `llama-cpp-2/src/context/params/get_set.rs` — `with_context_type` / `context_type`.
- `llama-cpp-2/src/speculative.rs` — new module (`MtpSpeculator`, errors).
- `llama-cpp-2/src/model.rs` — `new_mtp_context`.
- `llama-cpp-2/src/lib.rs` — declare `pub mod speculative;`.
- `examples/speculative-mtp/` — new crate (Cargo.toml + src/main.rs).
- root `Cargo.toml` — add the new example as a workspace member.

## Non-goals

- No EAGLE3 / n-gram speculative wiring (separate effort).
- No multi-sequence / batched-server orchestration (single-sequence loop; the shim takes
  `n_seq` so it can be generalized later).
- No C++ inference/graph changes — the upstream MTP implementation is used as-is.
- No `common_sampler_sample_and_accept_n` shim — acceptance stays in Rust.
