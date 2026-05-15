use crate::log_line::LogLine;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeOutput {
    None,
    Line(LogLine),
    TwoLines { earlier: LogLine, current: LogLine },
}
