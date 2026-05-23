//! Process-wide serialization for tests that need to initialize `LlamaBackend`.
//!
//! `LlamaBackend::init` is a once-per-process operation; concurrent attempts collide. Tests in
//! multiple modules each need access to a shared mutex so they take turns. This module exports
//! that shared mutex.

#[cfg(test)]
pub static BACKEND_INIT_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
