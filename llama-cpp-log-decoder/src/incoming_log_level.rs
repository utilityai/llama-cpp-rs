#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncomingLogLevel {
    Cont,
    Debug,
    Error,
    Info,
    None,
    Unknown(u32),
    Warn,
}

#[cfg(test)]
mod tests {
    use super::IncomingLogLevel;

    #[test]
    fn unknown_variant_equality() {
        assert_eq!(IncomingLogLevel::Unknown(42), IncomingLogLevel::Unknown(42));
        assert_ne!(IncomingLogLevel::Unknown(42), IncomingLogLevel::Unknown(43));
    }
}
