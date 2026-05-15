//! Decoder for the llama.cpp / ggml log callback stream.
//!
//! The C side delivers log lines in fragments: a missing trailing newline
//! signals that more fragments will follow at `GGML_LOG_LEVEL_CONT`. This
//! crate is a pure `&mut self` transducer — feed `(level, text)` pairs, get
//! complete [`LogLine`]s back when the trailing newline arrives. No globals,
//! no atomics, no FFI, no logger.
//!
//! [`LogLine`]: log_line::LogLine

pub mod decode_anomaly;
pub mod decode_output;
pub mod decode_result;
pub mod incoming_log_level;
pub mod log_decoder;
pub mod log_level;
pub mod log_line;
