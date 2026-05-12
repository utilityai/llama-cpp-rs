use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolCallArguments {
    ValidJson(Value),
    InvalidJson(String),
}

impl ToolCallArguments {
    #[must_use]
    pub fn from_string(raw: String) -> Self {
        serde_json::from_str::<Value>(&raw).map_or_else(|_| Self::InvalidJson(raw), Self::ValidJson)
    }
}

impl Default for ToolCallArguments {
    fn default() -> Self {
        Self::InvalidJson(String::new())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ToolCallArguments;

    #[test]
    fn from_string_object_returns_valid() {
        let result = ToolCallArguments::from_string(r#"{"location":"Paris"}"#.to_owned());

        assert_eq!(
            result,
            ToolCallArguments::ValidJson(json!({"location": "Paris"}))
        );
    }

    #[test]
    fn from_string_array_returns_valid() {
        let result = ToolCallArguments::from_string("[1,2,3]".to_owned());

        assert_eq!(result, ToolCallArguments::ValidJson(json!([1, 2, 3])));
    }

    #[test]
    fn from_string_scalar_returns_valid() {
        let result = ToolCallArguments::from_string("42".to_owned());

        assert_eq!(result, ToolCallArguments::ValidJson(json!(42)));
    }

    #[test]
    fn from_string_unparseable_returns_invalid() {
        let raw = "{not really json".to_owned();
        let result = ToolCallArguments::from_string(raw.clone());

        assert_eq!(result, ToolCallArguments::InvalidJson(raw));
    }

    #[test]
    fn from_string_empty_returns_invalid() {
        let result = ToolCallArguments::from_string(String::new());

        assert_eq!(result, ToolCallArguments::InvalidJson(String::new()));
    }

    #[test]
    fn default_is_empty_invalid() {
        assert_eq!(
            ToolCallArguments::default(),
            ToolCallArguments::InvalidJson(String::new())
        );
    }
}
