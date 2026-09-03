use core::fmt;
use std::ptr::NonNull;

#[repr(transparent)]
pub(crate) struct Ptr<T: ?Sized>(NonNull<T>);

impl<T: ?Sized> Ptr<T> {
    #[inline]
    pub(crate) fn new(ptr: *mut T) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    #[inline]
    pub(crate) const fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T: ?Sized> fmt::Debug for Ptr<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward to inner Debug
        fmt::Debug::fmt(&self.0, f)
    }
}
