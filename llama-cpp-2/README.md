# llama-cpp-rs-2

A wrapper around the [llama-cpp](https://github.com/ggerganov/llama.cpp/) library for rust.

# Goals

- Safe
- Up to date (llama.cpp moves fast)
- 100% API coverage (not yet complete)
- Abort free (llama.cpp will abort if you violate its invariants. This library will attempt to prevent that by ether
  ensuring the invariants are upheld statically or by checking them ourselves and returning an error)
- Performant (no meaningful overhead over using llama-cpp-sys-2)
- Well documented

# Non-goals

- Idiomatic rust (I will prioritize a more direct translation of the C++ API over a more idiomatic rust API due to
  maintenance burden)

# Contributing

Contributions are welcome. Please open an issue before starting work on a PR.