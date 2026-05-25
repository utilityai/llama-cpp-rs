use std::ffi::{CStr, CString};
use std::str::Utf8Error;

#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(pub CString);

impl LlamaChatTemplate {
    /// # Errors
    /// Returns an error if the template string contains null bytes.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    #[must_use]
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaChatTemplate;

    #[test]
    fn valid_template_creation() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let template_str = template.to_str().unwrap();

        assert_eq!(template_str, "chatml");
    }

    #[test]
    fn null_byte_returns_error() {
        let template = LlamaChatTemplate::new("null\0byte");

        assert!(template.is_err());
    }

    #[test]
    fn debug_formatting() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let debug_output = format!("{template:?}");

        assert!(debug_output.contains("chatml"));
    }

    #[test]
    fn to_string_returns_owned_string() {
        let template = LlamaChatTemplate::new("llama3").unwrap();
        let owned = template.to_string().unwrap();

        assert_eq!(owned, "llama3");
    }

    #[test]
    fn as_c_str_returns_valid_cstr() {
        let template = LlamaChatTemplate::new("test").unwrap();
        let cstr = template.as_c_str();

        assert_eq!(cstr.to_str().unwrap(), "test");
    }
}
