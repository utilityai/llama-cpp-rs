use crate::frame_stack;
use crate::recorded_error::RecordedError;

/// An RAII capture scope for errors raised inside FFI callbacks.
///
/// While an `ErrorScope` is alive, an error recorded via [`crate::record`] on
/// the same thread is captured in this scope's frame. Scopes nest: each
/// [`ErrorScope::enter`] pushes its own frame and [`Drop`] pops it, so an inner
/// FFI call cannot leak an error into an outer one.
#[derive(Debug)]
#[non_exhaustive]
pub struct ErrorScope;

impl ErrorScope {
    #[must_use]
    pub fn enter() -> Self {
        frame_stack::push_frame();

        Self
    }

    #[must_use]
    pub fn take(&self) -> Option<RecordedError> {
        frame_stack::take_from_top()
    }
}

impl Drop for ErrorScope {
    fn drop(&mut self) {
        frame_stack::pop_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorScope;
    use crate::record::record;
    use crate::recorded_error::RecordedError;

    #[test]
    fn records_and_takes_within_a_scope() {
        let scope = ErrorScope::enter();
        record(RecordedError::new("boom".to_string()));

        assert_eq!(
            scope.take().map(RecordedError::into_message),
            Some("boom".to_string())
        );
    }

    #[test]
    fn keeps_the_first_recorded_error() {
        let scope = ErrorScope::enter();
        record(RecordedError::new("first".to_string()));
        record(RecordedError::new("second".to_string()));

        assert_eq!(
            scope.take().map(RecordedError::into_message),
            Some("first".to_string())
        );
    }

    #[test]
    fn take_without_a_recorded_error_is_none() {
        let scope = ErrorScope::enter();

        assert!(scope.take().is_none());
    }

    #[test]
    fn take_consumes_the_recorded_error() {
        let scope = ErrorScope::enter();
        record(RecordedError::new("once".to_string()));

        assert!(scope.take().is_some());
        assert!(scope.take().is_none());
    }

    #[test]
    fn nested_scopes_capture_independently() {
        let outer = ErrorScope::enter();
        {
            let inner = ErrorScope::enter();
            record(RecordedError::new("inner".to_string()));

            assert_eq!(
                inner.take().map(RecordedError::into_message),
                Some("inner".to_string())
            );
        }

        record(RecordedError::new("outer".to_string()));

        assert_eq!(
            outer.take().map(RecordedError::into_message),
            Some("outer".to_string())
        );
    }

    #[test]
    fn recording_without_an_active_scope_is_dropped() {
        record(RecordedError::new("orphan".to_string()));

        let scope = ErrorScope::enter();

        assert!(scope.take().is_none());
    }
}
