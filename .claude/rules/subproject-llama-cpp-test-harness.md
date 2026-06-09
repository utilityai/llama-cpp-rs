---
paths:
  - "llama-cpp-test-harness/**"
  - "llama-cpp-test-harness-macros/**"
---

# `llama-cpp-test-harness` Context

- The purpose of `llama-cpp-test-harness` is to provide a custom harness that optimizes the tests to minimize model swaps.
- It must analyze all the relevant test attributes, and plan the execution to minimize the model swaps
- It needs to group the tests by model type they depend on, and execute them in phases (where each phase represents a different model)
