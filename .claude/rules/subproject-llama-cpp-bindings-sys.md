---
paths:
  - "llama-cpp-bindings-sys/**"
---

# `llama-cpp-bindings-sys` Context

- Every CPP exception MUST be surfaced to the Rust side of the project.
- If a CPP issue can be precisely identified, and mapped into an enum on the Rust side, it must be mapped.
- CPP bindings must remain minimal wrappers over `llama.cpp` API. Every logic possible must be moved to Rust, and be unit testable.
