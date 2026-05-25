//! Integration test fixtures for `llama-cpp-bindings`.
//!
//! This crate hosts test-only helpers used by the integration tests in `tests/`:
//! [`classify_sample_loop`] for sampling-loop drivers and [`test_model::fixtures_dir`]
//! for locating image fixtures.

pub mod classify_sample_loop;
pub mod test_model;
