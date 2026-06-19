# MTP Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind llama.cpp's already-implemented `draft-mtp` speculative decoding into the Rust crates so downstream code can run an MTP speculative loop with a single embedded-NextN GGUF.

**Architecture:** A flat `extern "C"` shim (`llama_rs_speculative_*`) in `llama-cpp-sys-2` wraps the C++ `common_speculative` API (libcommon, behind the default `common` feature). `llama-cpp-2` adds a `LlamaContextType` enum, an MTP draft-context constructor, and a safe `MtpSpeculator` that owns the shim handle. An example binary drives the speculative loop and doubles as a smoke test.

**Tech Stack:** Rust (llama-cpp-2 / llama-cpp-sys-2), C++ shim, bindgen, llama.cpp pinned at `9e3b928`, model `Qwen3.5-4B-UD-Q4_K_XL.gguf` (arch `qwen35`).

## Global Constraints

- llama.cpp pinned at commit `9e3b928`; do **not** bump the submodule.
- The shim is compiled only under the default cargo feature `common` (sets `-DLLAMA_RS_BUILD_COMMON`). No new feature flag.
- Bindgen already allowlists `llama_rs_.*` and `llama_.*`; no allowlist change needed.
- No C++ inference/graph changes — use upstream MTP as-is.
- Acceptance logic lives in Rust; do **not** shim `common_sampler_sample_and_accept_n`.
- `cargo build` of `llama-cpp-sys-2` compiles llama.cpp from source (minutes); build CPU-only first, GPU at smoke time.
- Reference for the loop ordering: `llama-cpp-sys-2/llama.cpp/tools/server/server-context.cpp` (MTP path) and `examples/speculative-simple/speculative-simple.cpp` (single-seq skeleton).

---

### Task 1: C shim `llama_rs_speculative_*`

**Files:**
- Modify: `llama-cpp-sys-2/wrapper_utils.h` (add `LLAMA_RS_STATUS_BUFFER_TOO_SMALL`)
- Modify: `llama-cpp-sys-2/wrapper_common.h` (declarations + opaque type)
- Modify: `llama-cpp-sys-2/wrapper_common.cpp` (implementations, include `speculative.h`)

**Interfaces:**
- Consumes: `common_speculative_*` from `common/speculative.h`, `common_params_speculative` from `common/common.h`, `llama_rs_status` from `wrapper_utils.h`.
- Produces (allowlisted as `llama_rs_.*`, consumed by Task 4):
  - `llama_rs_speculative * llama_rs_speculative_init_mtp(llama_context *ctx_tgt, llama_context *ctx_dft, int32_t n_max, int32_t n_min, float p_min, bool backend_sampling)`
  - `void llama_rs_speculative_free(llama_rs_speculative *)`
  - `int32_t llama_rs_speculative_n_max(const llama_rs_speculative *)`
  - `bool llama_rs_speculative_need_embd_nextn(const llama_rs_speculative *)`
  - `llama_rs_status llama_rs_speculative_begin(llama_rs_speculative *, llama_seq_id, const llama_token *prompt, size_t prompt_len)`
  - `llama_rs_status llama_rs_speculative_process(llama_rs_speculative *, const llama_batch *)`
  - `llama_rs_status llama_rs_speculative_draft(llama_rs_speculative *, llama_seq_id, llama_pos n_past, llama_token id_last, const llama_token *prompt, size_t prompt_len, llama_token *out_buf, size_t out_cap, size_t *out_len)`
  - `llama_rs_status llama_rs_speculative_accept(llama_rs_speculative *, llama_seq_id, uint16_t n_accepted)`

- [ ] **Step 1: Add status code.** In `wrapper_utils.h`, extend the enum:

```c
typedef enum llama_rs_status {
    LLAMA_RS_STATUS_OK = 0,
    LLAMA_RS_STATUS_INVALID_ARGUMENT = -1,
    LLAMA_RS_STATUS_ALLOCATION_FAILED = -2,
    LLAMA_RS_STATUS_EXCEPTION = -3,
    LLAMA_RS_STATUS_BUFFER_TOO_SMALL = -4
} llama_rs_status;
```

- [ ] **Step 2: Declare the API in `wrapper_common.h`** (inside the `extern "C"` block, after the existing declarations):

```c
// Opaque handle == common_speculative*
typedef struct llama_rs_speculative llama_rs_speculative;

llama_rs_speculative * llama_rs_speculative_init_mtp(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float   p_min,
    bool    backend_sampling);

void    llama_rs_speculative_free(llama_rs_speculative * spec);
int32_t llama_rs_speculative_n_max(const llama_rs_speculative * spec);
bool    llama_rs_speculative_need_embd_nextn(const llama_rs_speculative * spec);

llama_rs_status llama_rs_speculative_begin(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    const llama_token * prompt, size_t prompt_len);

llama_rs_status llama_rs_speculative_process(
    llama_rs_speculative * spec, const struct llama_batch * batch);

llama_rs_status llama_rs_speculative_draft(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last,
    const llama_token * prompt, size_t prompt_len,
    llama_token * out_buf, size_t out_cap, size_t * out_len);

llama_rs_status llama_rs_speculative_accept(
    llama_rs_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);
```

Add `#include <stdint.h>` if not already present.

- [ ] **Step 3: Implement in `wrapper_common.cpp`.** Add `#include "llama.cpp/common/speculative.h"` near the other includes, then append:

```cpp
static common_speculative * as_spec(llama_rs_speculative * h) {
    return reinterpret_cast<common_speculative *>(h);
}

extern "C" llama_rs_speculative * llama_rs_speculative_init_mtp(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float   p_min,
    bool    backend_sampling) {
    if (!ctx_tgt || !ctx_dft) {
        return nullptr;
    }
    try {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.ctx_tgt          = ctx_tgt;
        params.draft.ctx_dft          = ctx_dft;
        params.draft.n_max            = n_max;
        params.draft.n_min            = n_min;
        params.draft.p_min            = p_min;
        params.draft.backend_sampling = backend_sampling;
        common_speculative * spec = common_speculative_init(params, /*n_seq=*/ 1);
        return reinterpret_cast<llama_rs_speculative *>(spec);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_speculative_free(llama_rs_speculative * spec) {
    if (spec) {
        common_speculative_free(as_spec(spec));
    }
}

extern "C" int32_t llama_rs_speculative_n_max(const llama_rs_speculative * spec) {
    // we passed n_seq=1 and a single DRAFT_MTP type; n_max is bounded by params.draft.n_max,
    // which common_speculative clamps. Re-derive via the public helper is not available on the
    // handle, so we return the configured ceiling by querying the impl through a draft of size 0.
    // Simplest correct bound: ask common_speculative via its stored params is not exposed; instead
    // the Rust side stores n_max from init. This accessor exists for symmetry and returns -1 when
    // unknown.
    (void) spec;
    return -1;
}

extern "C" bool llama_rs_speculative_need_embd_nextn(const llama_rs_speculative * spec) {
    if (!spec) {
        return false;
    }
    return common_speculative_need_embd_nextn(as_spec(const_cast<llama_rs_speculative *>(spec)));
}

extern "C" llama_rs_status llama_rs_speculative_begin(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    const llama_token * prompt, size_t prompt_len) {
    if (!spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_tokens p;
        if (prompt && prompt_len) {
            p.assign(prompt, prompt + prompt_len);
        }
        common_speculative_begin(as_spec(spec), seq_id, p);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_process(
    llama_rs_speculative * spec, const struct llama_batch * batch) {
    if (!spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        const bool ok = common_speculative_process(as_spec(spec), *batch);
        return ok ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_draft(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last,
    const llama_token * prompt, size_t prompt_len,
    llama_token * out_buf, size_t out_cap, size_t * out_len) {
    if (!spec || !out_buf || !out_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_tokens prompt_vec;
        if (prompt && prompt_len) {
            prompt_vec.assign(prompt, prompt + prompt_len);
        }
        llama_tokens result_vec;

        common_speculative_draft_params & dp = common_speculative_get_draft_params(as_spec(spec), seq_id);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = n_past;
        dp.id_last  = id_last;
        dp.prompt   = &prompt_vec;
        dp.result   = &result_vec;

        common_speculative_draft(as_spec(spec));

        *out_len = result_vec.size();
        if (result_vec.size() > out_cap) {
            return LLAMA_RS_STATUS_BUFFER_TOO_SMALL;
        }
        if (!result_vec.empty()) {
            std::memcpy(out_buf, result_vec.data(), result_vec.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_accept(
    llama_rs_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    if (!spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        common_speculative_accept(as_spec(spec), seq_id, n_accepted);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
```

Note: the `llama_rs_speculative_n_max` body above is a stub returning -1 because `common_speculative` does not expose its configured n_max on the handle; the Rust wrapper stores n_max from init and uses that to size the draft buffer. Keep the accessor for ABI symmetry.

- [ ] **Step 4: Build the sys crate (CPU) to compile the shim + regenerate bindings.**

Run: `cargo build -p llama-cpp-sys-2 2>&1 | tail -20`
Expected: build succeeds (compiles llama.cpp + `wrapper_common.cpp`). This takes several minutes the first time.

- [ ] **Step 5: Verify bindgen emitted the symbols.**

Run: `grep -rl "llama_rs_speculative_init_mtp" target/*/build/llama-cpp-sys-2-*/out/bindings.rs | head`
Expected: at least one generated `bindings.rs` contains `llama_rs_speculative_init_mtp` and the other `llama_rs_speculative_*` functions.

- [ ] **Step 6: Commit.**

```bash
git add llama-cpp-sys-2/wrapper_utils.h llama-cpp-sys-2/wrapper_common.h llama-cpp-sys-2/wrapper_common.cpp
git commit -m "feat(sys): common_speculative MTP C shim (llama_rs_speculative_*)"
```

---

### Task 2: `LlamaContextType` enum + builder

**Files:**
- Modify: `llama-cpp-2/src/context/params.rs` (enum + re-export)
- Modify: `llama-cpp-2/src/context/params/get_set.rs` (`with_context_type`, `context_type`)
- Test: inline `#[cfg(test)]` in `params.rs`

**Interfaces:**
- Produces (consumed by Task 3): `LlamaContextType { Default, Mtp }`, `LlamaContextParams::with_context_type(self, LlamaContextType) -> Self`, `LlamaContextParams::context_type(&self) -> LlamaContextType`.

- [ ] **Step 1: Write the failing test.** Append to `llama-cpp-2/src/context/params.rs`:

```rust
#[cfg(test)]
mod context_type_tests {
    use super::{LlamaContextParams, LlamaContextType};

    #[test]
    fn context_type_round_trips() {
        let params = LlamaContextParams::default();
        assert_eq!(params.context_type(), LlamaContextType::Default);
        let params = params.with_context_type(LlamaContextType::Mtp);
        assert_eq!(params.context_type(), LlamaContextType::Mtp);
        assert_eq!(
            params.context_params.ctx_type,
            llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_MTP
        );
    }
}
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cargo test -p llama-cpp-2 --lib context_type_round_trips 2>&1 | tail -15`
Expected: FAIL — `LlamaContextType` / `with_context_type` / `context_type` not found.

- [ ] **Step 3: Add the enum.** In `params.rs`, near the other public enums, add:

```rust
/// The context type, mirroring `llama_context_type`. `Mtp` selects the
/// Multi-Token-Prediction (NextN) draft graph used for MTP speculative decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaContextType {
    /// Standard decoder context (`LLAMA_CONTEXT_TYPE_DEFAULT`).
    Default,
    /// MTP / NextN draft context (`LLAMA_CONTEXT_TYPE_MTP`).
    Mtp,
}

impl LlamaContextType {
    pub(crate) fn to_raw(self) -> llama_cpp_sys_2::llama_context_type {
        match self {
            Self::Default => llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_DEFAULT,
            Self::Mtp => llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_MTP,
        }
    }

    pub(crate) fn from_raw(raw: llama_cpp_sys_2::llama_context_type) -> Self {
        match raw {
            llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_MTP => Self::Mtp,
            _ => Self::Default,
        }
    }
}
```

- [ ] **Step 4: Add the builder/getter.** In `get_set.rs`, inside `impl LlamaContextParams`, add (import `LlamaContextType` in the `use super::{...}` line):

```rust
    /// Set the context type (e.g. [`LlamaContextType::Mtp`] for MTP speculative decoding).
    #[must_use]
    pub fn with_context_type(mut self, ctx_type: LlamaContextType) -> Self {
        self.context_params.ctx_type = ctx_type.to_raw();
        self
    }

    /// Get the context type.
    #[must_use]
    pub fn context_type(&self) -> LlamaContextType {
        LlamaContextType::from_raw(self.context_params.ctx_type)
    }
```

- [ ] **Step 5: Run to verify it passes.**

Run: `cargo test -p llama-cpp-2 --lib context_type_round_trips 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add llama-cpp-2/src/context/params.rs llama-cpp-2/src/context/params/get_set.rs
git commit -m "feat(cpp-2): LlamaContextType + with_context_type builder"
```

---

### Task 3: MTP draft-context constructor

**Files:**
- Create: `llama-cpp-2/src/speculative.rs` (errors + constructor lives here; speculator added in Task 4)
- Modify: `llama-cpp-2/src/lib.rs` (`pub mod speculative;`)

**Interfaces:**
- Consumes: `LlamaContext.context` (`pub(crate) NonNull<llama_context>`), `LlamaContext::new` (`pub(crate)`), `LlamaModel`, `LlamaContextParams`, `LlamaContextType` (Task 2).
- Produces (consumed by example): `LlamaModel::new_mtp_context<'a>(&'a self, &LlamaBackend, target: &'a LlamaContext<'a>, params: LlamaContextParams) -> Result<LlamaContext<'a>, SpeculativeError>`, and `SpeculativeError`.

- [ ] **Step 1: Create `speculative.rs` with the error type and constructor.**

```rust
//! MTP (Multi-Token Prediction / NextN) speculative decoding.

use std::ptr::NonNull;

use crate::context::params::{LlamaContextParams, LlamaContextType};
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::LlamaModel;

/// Errors from MTP speculative decoding.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeError {
    /// A shim call rejected its arguments.
    #[error("invalid argument passed to speculative shim")]
    InvalidArgument,
    /// `llama_decode` failed inside the speculative step.
    #[error("decode failed inside speculative step")]
    DecodeFailed,
    /// The draft output buffer was too small.
    #[error("draft buffer too small (needed {needed}, had {had})")]
    BufferTooSmall {
        /// tokens the draft produced
        needed: usize,
        /// buffer capacity
        had: usize,
    },
    /// A C++ exception was caught at the shim boundary.
    #[error("c++ exception at speculative shim boundary")]
    Exception,
    /// `common_speculative_init` returned null.
    #[error("failed to initialize speculative context")]
    InitFailed,
    /// Creating the MTP draft context failed.
    #[error("failed to create MTP draft context")]
    ContextCreationFailed,
    /// Target/draft embedding widths disagree.
    #[error("n_embd mismatch between target and draft contexts")]
    EmbdMismatch,
}

pub(crate) fn status_to_result(status: i32) -> Result<(), SpeculativeError> {
    match status {
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_OK => Ok(()),
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(SpeculativeError::InvalidArgument)
        }
        x if x == llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => Err(SpeculativeError::Exception),
        _ => Err(SpeculativeError::Exception),
    }
}

impl LlamaModel {
    /// Create the MTP draft context for `target`. The draft context runs the NextN graph on the
    /// **same model** and shares memory with `target` (`ctx_type = Mtp`, `ctx_other = target`).
    /// Mirrors `tools/server/server-context.cpp` MTP setup.
    pub fn new_mtp_context<'a>(
        &'a self,
        _: &LlamaBackend,
        target: &'a LlamaContext<'a>,
        mut params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, SpeculativeError> {
        params.context_params.ctx_type = LlamaContextType::Mtp.to_raw();
        params.context_params.ctx_other = target.context.as_ptr();

        let ctx = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(
                self.model.as_ptr(),
                params.context_params,
            )
        };
        let ctx = NonNull::new(ctx).ok_or(SpeculativeError::ContextCreationFailed)?;
        Ok(LlamaContext::new(self, ctx, params.embeddings()))
    }
}
```

Note: `to_raw` is `pub(crate)`; if the privacy is wrong adjust the `use`. Confirm `LlamaModel.model` field is reachable (it is `pub(crate)` — used by `new_context`). Confirm `LlamaContextParams::embeddings()` exists (used by `new_context`).

- [ ] **Step 2: Register the module.** In `lib.rs`, add `pub mod speculative;` alongside the other `pub mod` lines.

- [ ] **Step 3: Build to typecheck.**

Run: `cargo build -p llama-cpp-2 2>&1 | tail -20`
Expected: builds. Fix privacy (`pub(crate)`) issues on `to_raw`/`from_raw`/field access if the compiler flags them.

- [ ] **Step 4: Commit.**

```bash
git add llama-cpp-2/src/speculative.rs llama-cpp-2/src/lib.rs
git commit -m "feat(cpp-2): SpeculativeError + new_mtp_context"
```

---

### Task 4: `MtpSpeculator` safe wrapper

**Files:**
- Modify: `llama-cpp-2/src/speculative.rs` (add `MtpSpeculator`)
- Test: inline `#[cfg(test)]` for `status_to_result` mapping

**Interfaces:**
- Consumes: shim from Task 1, `LlamaContext`, `LlamaBatch.llama_batch` (`pub(crate)`), `LlamaToken` (`#[repr(transparent)]` over `i32`).
- Produces (consumed by example): `MtpSpeculator::new`, `.begin`, `.process`, `.draft`, `.accept`, `.need_embd_nextn`.

- [ ] **Step 1: Write the failing test** (append to `speculative.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::{status_to_result, SpeculativeError};

    #[test]
    fn status_mapping() {
        assert!(status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_OK).is_ok());
        assert!(matches!(
            status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT),
            Err(SpeculativeError::InvalidArgument)
        ));
        assert!(matches!(
            status_to_result(llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION),
            Err(SpeculativeError::Exception)
        ));
    }
}
```

- [ ] **Step 2: Run to verify it fails (or passes trivially) — confirm it compiles against the constants.**

Run: `cargo test -p llama-cpp-2 --lib speculative::tests::status_mapping 2>&1 | tail -15`
Expected: PASS once `status_to_result` from Task 3 is present (this test guards future regressions and confirms the sys constants exist).

- [ ] **Step 3: Add `MtpSpeculator`** to `speculative.rs`:

```rust
use crate::llama_batch::LlamaBatch;
use crate::token::LlamaToken;

/// A safe handle to an MTP speculator. Borrows the target and draft contexts.
#[derive(Debug)]
pub struct MtpSpeculator<'a> {
    handle: NonNull<llama_cpp_sys_2::llama_rs_speculative>,
    n_max: i32,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> MtpSpeculator<'a> {
    /// Initialize an MTP speculator over `target` (real model) and `draft` (MTP context from
    /// [`LlamaModel::new_mtp_context`]).
    pub fn new(
        target: &'a LlamaContext<'a>,
        draft: &'a LlamaContext<'a>,
        n_max: i32,
        n_min: i32,
        p_min: f32,
        backend_sampling: bool,
    ) -> Result<Self, SpeculativeError> {
        let handle = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_init_mtp(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                n_max,
                n_min,
                p_min,
                backend_sampling,
            )
        };
        let handle = NonNull::new(handle).ok_or(SpeculativeError::InitFailed)?;
        Ok(Self {
            handle,
            n_max,
            _marker: std::marker::PhantomData,
        })
    }

    /// True if the speculator needs target NextN embeddings extracted.
    #[must_use]
    pub fn need_embd_nextn(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_need_embd_nextn(self.handle.as_ptr()) }
    }

    /// Optionally call once at the start of a generation with the prompt.
    pub fn begin(&mut self, seq_id: i32, prompt: &[LlamaToken]) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_begin(
                self.handle.as_ptr(),
                seq_id,
                prompt.as_ptr().cast(),
                prompt.len(),
            )
        };
        status_to_result(status)
    }

    /// Feed the just-decoded target verify batch so the speculator captures hidden states.
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_process(
                self.handle.as_ptr(),
                std::ptr::addr_of!(batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Generate draft tokens for `seq_id`.
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
        prompt: &[LlamaToken],
    ) -> Result<Vec<LlamaToken>, SpeculativeError> {
        let cap = self.n_max.max(1) as usize;
        let mut buf = vec![LlamaToken(0); cap];
        let mut out_len: usize = 0;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_draft(
                self.handle.as_ptr(),
                seq_id,
                n_past,
                id_last.0,
                prompt.as_ptr().cast(),
                prompt.len(),
                buf.as_mut_ptr().cast(),
                buf.len(),
                std::ptr::addr_of_mut!(out_len),
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_BUFFER_TOO_SMALL {
            return Err(SpeculativeError::BufferTooSmall {
                needed: out_len,
                had: cap,
            });
        }
        status_to_result(status)?;
        buf.truncate(out_len);
        Ok(buf)
    }

    /// Inform the speculator that `n_accepted` draft tokens were accepted.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_accept(self.handle.as_ptr(), seq_id, n_accepted)
        };
        status_to_result(status)
    }
}

impl Drop for MtpSpeculator<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_free(self.handle.as_ptr()) }
    }
}
```

- [ ] **Step 4: Build + run unit tests.**

Run: `cargo test -p llama-cpp-2 --lib speculative 2>&1 | tail -20`
Expected: compiles; `status_mapping` PASS.

- [ ] **Step 5: Commit.**

```bash
git add llama-cpp-2/src/speculative.rs
git commit -m "feat(cpp-2): MtpSpeculator safe wrapper over the shim"
```

---

### Task 5: `examples/speculative-mtp` (loop + smoke test)

**Files:**
- Create: `examples/speculative-mtp/Cargo.toml`
- Create: `examples/speculative-mtp/src/main.rs`
- Create: `examples/speculative-mtp/README.md` (per-backend probe commands)
- Modify: root `Cargo.toml` (add workspace member)

**Interfaces:**
- Consumes: `LlamaModel::new_mtp_context`, `MtpSpeculator`, `LlamaBatch`, `LlamaContext::{decode,get_logits_ith}` from earlier tasks.
- Produces: a binary `speculative-mtp` that prints generated text and `accept%`.

- [ ] **Step 1: Add the workspace member.** In root `Cargo.toml` `members = [...]`, add `"examples/speculative-mtp",`.

- [ ] **Step 2: Create `examples/speculative-mtp/Cargo.toml`** (mirror `examples/simple/Cargo.toml`; backend features forwarded to `llama-cpp-2`):

```toml
[package]
name = "speculative-mtp"
version = "0.1.0"
edition = "2021"

[dependencies]
llama-cpp-2 = { path = "../../llama-cpp-2" }
clap = { workspace = true, features = ["derive"] }
anyhow = { workspace = true }

[features]
cuda = ["llama-cpp-2/cuda"]
metal = ["llama-cpp-2/metal"]
vulkan = ["llama-cpp-2/vulkan"]
```

- [ ] **Step 3: Write `src/main.rs`.** The loop ports `speculative-simple.cpp` for a single sequence, greedy acceptance in Rust, with the MTP `process()` call inserted after each target decode. Read `tools/server/server-context.cpp` MTP path (search `DRAFT_MTP`, `need_embd_nextn`, `common_speculative_process`) and confirm the prefill→process→draft ordering before finalizing.

```rust
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::speculative::MtpSpeculator;
use llama_cpp_2::token::LlamaToken;

#[derive(Parser)]
struct Args {
    /// Path to the MTP GGUF (embedded NextN head).
    #[arg(long)]
    model: PathBuf,
    /// Prompt.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// Max tokens to generate.
    #[arg(long, default_value_t = 64)]
    n_predict: i32,
    /// Max draft tokens per step.
    #[arg(long, default_value_t = 4)]
    n_draft: i32,
    /// GPU layers (0 = CPU).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .context("load model")?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(4096));
    let mut ctx_tgt = model.new_context(&backend, ctx_params.clone())?;
    let ctx_dft = model.new_mtp_context(&backend, &ctx_tgt, ctx_params)?;

    let tokens = model.str_to_token(&args.prompt, AddBos::Always)?;
    let n_prompt = tokens.len() as i32;

    let mut spec = MtpSpeculator::new(&ctx_tgt, &ctx_dft, args.n_draft, 0, 0.0, true)?;

    // Prefill the target on all-but-last token.
    let mut batch = LlamaBatch::new(4096, 1);
    let prompt_head = &tokens[..tokens.len() - 1];
    for (i, t) in prompt_head.iter().enumerate() {
        batch.add(*t, i as i32, &[0], i == prompt_head.len() - 1)?;
    }
    ctx_tgt.decode(&mut batch)?;
    spec.process(&batch)?;
    spec.begin(0, prompt_head)?;

    let mut id_last = *tokens.last().unwrap();
    let mut n_past = n_prompt - 1;
    let mut prompt_tgt: Vec<LlamaToken> = prompt_head.to_vec();

    let mut n_drafted = 0i64;
    let mut n_accept = 0i64;
    print!("{}", args.prompt);
    std::io::stdout().flush().ok();

    while n_past < n_prompt + args.n_predict {
        let draft = spec.draft(0, n_past, id_last, &prompt_tgt)?;
        n_drafted += draft.len() as i64;

        // verify batch: [id_last, draft...]
        batch.clear();
        batch.add(id_last, n_past, &[0], true)?;
        for (i, d) in draft.iter().enumerate() {
            batch.add(*d, n_past + 1 + i as i32, &[0], true)?;
        }
        ctx_tgt.decode(&mut batch)?;
        spec.process(&batch)?;

        // greedy accept in Rust
        let mut ids: Vec<i32> = Vec::new();
        let mut i = 0usize;
        loop {
            let t = argmax(ctx_tgt.get_logits_ith(i as i32));
            ids.push(t);
            if i == draft.len() || t != draft[i].0 {
                break;
            }
            i += 1;
        }
        let n_acc = (ids.len() - 1) as u16;
        n_accept += n_acc as i64;
        spec.accept(0, n_acc)?;

        for &t in &ids {
            let tok = LlamaToken(t);
            if model.is_eog_token(tok) {
                n_past = n_prompt + args.n_predict; // force loop exit
                break;
            }
            let piece = model.token_to_str(tok, Special::Tokenize).unwrap_or_default();
            print!("{piece}");
            std::io::stdout().flush().ok();
            prompt_tgt.push(id_last);
            id_last = tok;
            n_past += 1;
        }
    }

    let pct = if n_drafted > 0 {
        100.0 * n_accept as f64 / n_drafted as f64
    } else {
        0.0
    };
    println!("\n\n[mtp] n_drafted={n_drafted} n_accept={n_accept} accept={pct:.1}%");
    Ok(())
}
```

Note: confirm exact method names during execution (`str_to_token`/`token_to_str`/`is_eog_token`/`LlamaBatch::new`/`with_n_gpu_layers`) against `model.rs` and `llama_batch.rs`; adjust call sites to the real signatures. The accept/draft/process ordering is the contract from `speculative-simple.cpp` + the server MTP path.

- [ ] **Step 4: Build the example (CPU).**

Run: `cargo build -p speculative-mtp 2>&1 | tail -20`
Expected: builds. Fix any API-name mismatches flagged.

- [ ] **Step 5: Commit.**

```bash
git add examples/speculative-mtp Cargo.toml
git commit -m "feat(examples): speculative-mtp example + smoke harness"
```

---

### Task 6: Smoke tests (CPU, CUDA, Metal)

**Files:**
- Create: `examples/speculative-mtp/README.md` (probe commands; created in Task 5 step or here)

- [ ] **Step 1: CPU smoke (correctness).**

Run:
```bash
cargo run -p speculative-mtp --release -- \
  --model /home/replikeit/Codes/lakeside/edge-ai/research/benchmarks/models/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" --n-predict 48 --n-gpu-layers 0
```
Expected: coherent continuation; final line shows `accept%` > 0.

- [ ] **Step 2: CUDA smoke (GPU path).**

Run:
```bash
cargo run -p speculative-mtp --release --features cuda -- \
  --model <same path> --prompt "Explain speculative decoding in one paragraph." \
  --n-predict 96 --n-gpu-layers 999
```
Expected: runs on the RTX 4080; coherent output; `accept%` > 0. If a backend op is missing, ggml logs a CPU fallback (still correct).

- [ ] **Step 3: Metal smoke (via SSH).** Sync the repo + model to `macbook-local`, build with `--features metal`, run the same command. Capture output. Record `accept%` and any backend warnings.

- [ ] **Step 4: Document probes** in `examples/speculative-mtp/README.md` (CPU/CUDA/Metal/Vulkan one-liners + expected `accept%` line) and commit.

```bash
git add examples/speculative-mtp/README.md
git commit -m "docs(examples): speculative-mtp per-backend smoke probes"
```

---

## Self-Review

**Spec coverage:** shim (Task 1) ✓; `LlamaContextType` + builder (Task 2) ✓; `new_mtp_context` (Task 3) ✓; `MtpSpeculator` + errors (Task 4) ✓; example/smoke (Task 5) ✓; CPU/CUDA/Metal smoke (Task 6) ✓; unit tests for enum round-trip (Task 2) and status mapping (Task 4) ✓.

**Placeholder scan:** the `llama_rs_speculative_n_max` C body is intentionally a `-1` stub (documented; Rust stores n_max from init). All other steps contain concrete code. Method-name confirmation notes in Tasks 3/5 are verification steps, not deferred work.

**Type consistency:** `MtpSpeculator` methods (`begin/process/draft/accept`) match the example call sites; `status_to_result`, `to_raw`/`from_raw`, `LlamaContextType` names consistent across tasks; shim signatures in Task 1 match the `unsafe` call sites in Task 4.

**Known risk to resolve during execution:** the exact prefill→`process`→`begin`→`draft` ordering and whether `ctx_dft` needs an explicit prefill decode under shared memory — resolve by reading `tools/server/server-context.cpp` MTP path before finalizing Task 5's loop.
