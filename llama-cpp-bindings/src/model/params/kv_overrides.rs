//! Key-value overrides for a model.

use crate::model::params::LlamaModelParams;
use crate::model::params::param_override_value::ParamOverrideValue;
use std::ffi::{CStr, CString};
use std::fmt::Debug;

/// A struct implementing [`IntoIterator`] over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrides<'model_params> {
    model_params: &'model_params LlamaModelParams,
}

impl KvOverrides<'_> {
    /// Creates a new `KvOverrides` view over the given model parameters.
    #[must_use]
    pub const fn new(model_params: &LlamaModelParams) -> KvOverrides<'_> {
        KvOverrides { model_params }
    }
}

impl<'model_params> IntoIterator for KvOverrides<'model_params> {
    type Item = (CString, ParamOverrideValue);
    type IntoIter = KvOverrideValueIterator<'model_params>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator {
            model_params: self.model_params,
            current: 0,
        }
    }
}

/// An iterator over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrideValueIterator<'model_params> {
    model_params: &'model_params LlamaModelParams,
    current: usize,
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = (CString, ParamOverrideValue);

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.params.kv_overrides;

        if overrides.is_null() {
            return None;
        }

        loop {
            // SAFETY: llama.cpp guarantees the last element contains an empty key.
            // We've checked the previous one in the last iteration, the next one
            // should be valid or 0 (and thus safe to deref).
            let current = unsafe { *overrides.add(self.current) };

            if current.key[0] == 0 {
                return None;
            }

            self.current += 1;

            if let Ok(value) = ParamOverrideValue::try_from(&current) {
                let key = unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() };

                return Some((key, value));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::pin::pin;

    use crate::model::params::LlamaModelParams;
    use crate::model::params::param_override_value::ParamOverrideValue;

    #[test]
    fn kv_overrides_empty_by_default() {
        let params = LlamaModelParams::default();
        let overrides = params.kv_overrides();
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }

    #[test]
    fn kv_overrides_iterates_single_entry() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("test_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(42))
            .unwrap();

        let entries: Vec<_> = params.kv_overrides().into_iter().collect();

        assert_eq!(entries.len(), 1);
        let (entry_key, entry_value) = &entries[0];
        assert_eq!(entry_key.to_bytes(), b"test_key");
        assert_eq!(*entry_value, ParamOverrideValue::Int(42));
    }

    #[test]
    fn kv_overrides_new_creates_view() {
        let params = LlamaModelParams::default();
        let overrides = super::KvOverrides::new(&params);
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }

    #[test]
    fn kv_overrides_skips_entry_with_unknown_tag() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("valid_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(99))
            .unwrap();

        params.kv_overrides[0].tag = 9999;

        assert_eq!(params.kv_overrides().into_iter().count(), 0);
    }
}
