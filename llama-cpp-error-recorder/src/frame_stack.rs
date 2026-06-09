use std::cell::RefCell;

use crate::recorded_error::RecordedError;

thread_local! {
    static FRAMES: RefCell<Vec<Option<RecordedError>>> = const { RefCell::new(Vec::new()) };
}

pub fn push_frame() {
    FRAMES.with(|cell| cell.borrow_mut().push(None));
}

pub fn pop_frame() {
    FRAMES.with(|cell| {
        cell.borrow_mut().pop();
    });
}

pub fn take_from_top() -> Option<RecordedError> {
    FRAMES.with(|cell| cell.borrow_mut().last_mut().and_then(Option::take))
}

pub fn record_into_top(error: RecordedError) {
    FRAMES.with(|cell| {
        let mut frames = cell.borrow_mut();
        if let Some(top) = frames.last_mut()
            && top.is_none()
        {
            *top = Some(error);
        }
    });
}
