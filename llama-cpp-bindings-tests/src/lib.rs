//! Integration test fixtures for `llama-cpp-bindings`.
//!
//! This crate is the only place in the workspace that loads model files. It
//! exists so production code in `llama-cpp-bindings` stays free of test-only
//! dependencies (`anyhow`, `hf-hub`, `serial_test`, …) and helpers.

pub mod classify_sample_loop;
pub mod fixture_session;
pub mod gpu_backend;
pub mod test_model;

pub use fixture_session::FixtureSession;
