use crate::log_level::LogLevel;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogLine {
    pub level: LogLevel,
    pub text: String,
}
