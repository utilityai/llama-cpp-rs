//! an rusty equivalent of `llama_token_data`.
use crate::token::data::LlamaTokenData;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// is the data sorted?
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and weather or not the data is sorted.
    ///
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self { data, sorted }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and weather or not the data is sorted.
    /// ```
    /// # use llama_cpp_2::token::data::LlamaTokenData;
    /// # use llama_cpp_2::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_2::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<T>(data: T, sorted: bool) -> LlamaTokenDataArray
    where
        T: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }
}

impl LlamaTokenDataArray {
    /// Modify the underlying data as a `llama_token_data_array`. and reconstruct the `LlamaTokenDataArray`.
    ///
    /// # Panics
    ///
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at runtime so breaking them is UB)
    ///
    /// SAFETY:
    /// [modify] cannot change the data pointer.
    /// if the data is not sorted, sorted must be false.
    /// the size of the data can only decrease (i.e you cannot add new elements).
    pub(crate) unsafe fn modify_as_c_llama_token_data_array(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_sys_2::llama_token_data_array),
    ) {
        let size = self.data.len();
        let data = self.data.as_mut_ptr().cast();
        let mut c_llama_token_data_array = llama_cpp_sys_2::llama_token_data_array {
            data,
            size,
            sorted: self.sorted,
        };
        modify(&mut c_llama_token_data_array);
        assert!(
            std::ptr::eq(data, c_llama_token_data_array.data),
            "data pointer changed"
        );
        assert!(c_llama_token_data_array.size <= size, "size increased");
        self.data.set_len(c_llama_token_data_array.size);
        self.sorted = c_llama_token_data_array.sorted;
    }
}
