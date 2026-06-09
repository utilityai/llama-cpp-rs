#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod frame_stack;

pub mod error_scope;
pub mod record;
pub mod recorded_error;

pub use error_scope::ErrorScope;
pub use record::record;
pub use recorded_error::RecordedError;
