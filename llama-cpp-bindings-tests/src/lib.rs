//! Integration tests for `llama-cpp-bindings`.
//!
//! `LlamaFixture::build_context` and `fixtures_dir` live in the
//! `llama-cpp-test-harness` crate. Helpers below stay here because their
//! error paths are unreachable from controlled fixture inputs and would
//! otherwise drag the harness's coverage gate down.

pub mod classify_sample_loop;
pub mod decode_hello_world;
