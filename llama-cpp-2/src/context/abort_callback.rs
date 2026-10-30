use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

// Sized wrapper so C gets a thin pointer to the boxed closure.
pub(super) struct AbortCallback(Box<dyn Fn() -> bool + Send + Sync + 'static>);

impl AbortCallback {
    pub(super) fn new(callback: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self(Box::new(callback))
    }
}

extern "C" fn trampoline(data: *mut c_void) -> bool {
    // SAFETY: `data` points into the context's boxed callback, which outlives every
    // native call: replacement synchronizes first, and `Drop` frees the context first.
    let callback = unsafe { &*data.cast::<AbortCallback>() };
    match catch_unwind(AssertUnwindSafe(|| (callback.0)())) {
        Ok(abort) => abort,
        Err(payload) => {
            // Dropping the payload could panic again across the FFI boundary.
            std::mem::forget(payload);
            true
        }
    }
}

// Registers the new pointer before dropping the old allocation; callers synchronize first.
pub(super) fn replace(
    current: &mut Option<Box<AbortCallback>>,
    next: Option<Box<AbortCallback>>,
    register: impl FnOnce(llama_cpp_sys_2::ggml_abort_callback, *mut c_void),
) {
    let (callback, data): (llama_cpp_sys_2::ggml_abort_callback, *mut c_void) =
        match next.as_deref() {
            Some(callback) => (
                Some(trampoline),
                std::ptr::from_ref(callback).cast_mut().cast(),
            ),
            None => (None, std::ptr::null_mut()),
        };
    register(callback, data);
    drop(std::mem::replace(current, next));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn trampoline_returns_callback_result() {
        for expected in [false, true] {
            let mut callback = AbortCallback::new(move || expected);
            assert_eq!(
                trampoline(std::ptr::from_mut(&mut callback).cast()),
                expected
            );
        }
    }

    #[test]
    fn trampoline_treats_panics_as_abort() {
        struct PanickingPayload;
        impl Drop for PanickingPayload {
            fn drop(&mut self) {
                panic!("panic payload must not be dropped across FFI");
            }
        }

        let mut callback = AbortCallback::new(|| std::panic::panic_any(PanickingPayload));
        assert!(trampoline(std::ptr::from_mut(&mut callback).cast()));
    }

    #[test]
    fn replacement_and_clear_unregister_before_drop() {
        struct CountDrop(Arc<AtomicUsize>);
        impl Drop for CountDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let make_callback = |result| {
            let guard = CountDrop(Arc::clone(&drops));
            Box::new(AbortCallback::new(move || {
                let _keep_alive = &guard;
                result
            }))
        };
        let mut current = None;
        let mut native = (None, std::ptr::null_mut());
        replace(&mut current, Some(make_callback(false)), |cb, data| {
            native = (cb, data);
        });
        assert!(!unsafe { native.0.unwrap()(native.1) });

        let old_data = native.1;
        replace(&mut current, Some(make_callback(true)), |cb, data| {
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert!(!unsafe { native.0.unwrap()(old_data) });
            assert_ne!(data, old_data);
            native = (cb, data);
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(unsafe { native.0.unwrap()(native.1) });

        replace(&mut current, None, |cb, data| {
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert!(unsafe { native.0.unwrap()(native.1) });
            native = (cb, data);
        });
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(native.0.is_none());
        assert!(native.1.is_null());
        assert!(current.is_none());

        replace(&mut current, None, |cb, data| {
            assert!(cb.is_none());
            assert!(data.is_null());
        });
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
