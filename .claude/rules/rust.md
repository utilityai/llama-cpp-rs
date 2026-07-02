---
paths:
  - "**/*.rs"
  - "**/Cargo.toml"
---

# Rust Standards

- Always use explicit lifetime variable names (do not use `'a` and such, use descriptive names like `'message` or similar)
- Always use descriptive parameter names (never use single letter names for generics
- Each file must contain at most a single struct, or single enum, or a single public function (at most one of any of those).
- Each file must contain at most a single public item. You can still keep multiple private function helpers. Files need to be named after their public item.
- Always destructure structs in arguments if possible.

# Code Style

Imports/uses must not be mixed with other kinds of rust syntax. 

Each file needs to follow this order: 
1. `pub mod`/`mod` exports 
2. vendor crate `use` 
3. project crate `use` 
4. local crate `use` 
5. private function helpers
6. private struct helpers
7. single public export
