use std::ffi::{CStr, CString};
use std::fmt::Debug;

use crate::model::params::LlamaModelParams;
use crate::model::params::param_override_value::ParamOverrideValue;

/// An iterator over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrideValueIterator<'model_params> {
    model_params: &'model_params LlamaModelParams,
    current: usize,
}

impl<'model_params> KvOverrideValueIterator<'model_params> {
    #[must_use]
    pub const fn new(model_params: &'model_params LlamaModelParams) -> Self {
        Self {
            model_params,
            current: 0,
        }
    }
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
