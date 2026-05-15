---
paths:
  - "llama-cpp-log-decoder/**"
---

# `llama-cpp-log-decoder` Standards

- The logging subsystem MUST NEVER panic, crash, or otherwise interrupt the program.
- Logs report issues; they must not cause them.
- No .unwrap(), .expect(), panic!(), or panic-prone indexing.
- No panic-prone slicing.
