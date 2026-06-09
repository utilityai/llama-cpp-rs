use crate::frame_stack;
use crate::recorded_error::RecordedError;

/// Records an error raised inside an FFI callback so the Rust code that drove
/// the FFI call can surface it via [`crate::error_scope::ErrorScope::take`].
///
/// Only the first error recorded in the active scope is kept. If no scope is
/// active the error is dropped: recording runs inside an FFI callback, where
/// unwinding is undefined behaviour, so it must never panic.
pub fn record(error: RecordedError) {
    frame_stack::record_into_top(error);
}
