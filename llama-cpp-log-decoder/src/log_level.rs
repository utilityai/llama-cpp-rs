#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogLevel {
    Debug,
    Error,
    Info,
    None,
    Unknown(u32),
    Warn,
}

#[cfg(test)]
mod tests {
    use super::LogLevel;

    #[test]
    fn unknown_variant_equality() {
        assert_eq!(LogLevel::Unknown(42), LogLevel::Unknown(42));
        assert_ne!(LogLevel::Unknown(42), LogLevel::Unknown(43));
    }
}
