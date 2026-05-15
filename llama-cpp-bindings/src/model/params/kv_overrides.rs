//! Key-value overrides for a model.

use std::fmt::Debug;

use crate::model::params::LlamaModelParams;
use crate::model::params::kv_override_value_iterator::KvOverrideValueIterator;

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
    type Item = <KvOverrideValueIterator<'model_params> as Iterator>::Item;
    type IntoIter = KvOverrideValueIterator<'model_params>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator::new(self.model_params)
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
