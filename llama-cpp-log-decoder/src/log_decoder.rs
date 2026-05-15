use crate::decode_anomaly::DecodeAnomaly;
use crate::decode_output::DecodeOutput;
use crate::decode_result::DecodeResult;
use crate::incoming_log_level::IncomingLogLevel;
use crate::log_level::LogLevel;
use crate::log_line::LogLine;

pub struct LogDecoder {
    buffered: Option<(LogLevel, String)>,
    previous_level: LogLevel,
}

impl LogDecoder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffered: None,
            previous_level: LogLevel::None,
        }
    }

    pub fn feed(&mut self, level: IncomingLogLevel, text: &str) -> DecodeResult {
        match level {
            IncomingLogLevel::Cont => self.feed_cont(text),
            IncomingLogLevel::Debug => self.feed_non_cont(LogLevel::Debug, text),
            IncomingLogLevel::Error => self.feed_non_cont(LogLevel::Error, text),
            IncomingLogLevel::Info => self.feed_non_cont(LogLevel::Info, text),
            IncomingLogLevel::None => self.feed_non_cont(LogLevel::None, text),
            IncomingLogLevel::Unknown(raw) => self.feed_non_cont(LogLevel::Unknown(raw), text),
            IncomingLogLevel::Warn => self.feed_non_cont(LogLevel::Warn, text),
        }
    }

    fn feed_cont(&mut self, text: &str) -> DecodeResult {
        if let Some((level, mut buffer)) = self.buffered.take() {
            buffer.push_str(text);
            if let Some(without_newline) = buffer.strip_suffix('\n') {
                DecodeResult {
                    output: DecodeOutput::Line(LogLine {
                        level,
                        text: without_newline.to_owned(),
                    }),
                    anomaly: None,
                }
            } else {
                self.buffered = Some((level, buffer));
                DecodeResult {
                    output: DecodeOutput::None,
                    anomaly: None,
                }
            }
        } else {
            self.feed_orphan_cont(text)
        }
    }

    fn feed_orphan_cont(&mut self, text: &str) -> DecodeResult {
        let level = self.previous_level;
        if let Some(without_newline) = text.strip_suffix('\n') {
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level,
                    text: without_newline.to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        } else {
            self.buffered = Some((level, text.to_owned()));
            DecodeResult {
                output: DecodeOutput::None,
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        }
    }

    fn feed_non_cont(&mut self, level: LogLevel, text: &str) -> DecodeResult {
        self.previous_level = level;
        let stale = self.buffered.take();
        match (text.strip_suffix('\n'), stale) {
            (Some(without_newline), Some((stale_level, stale_text))) => DecodeResult {
                output: DecodeOutput::TwoLines {
                    earlier: LogLine {
                        level: stale_level,
                        text: stale_text,
                    },
                    current: LogLine {
                        level,
                        text: without_newline.to_owned(),
                    },
                },
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            },
            (Some(without_newline), None) => DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level,
                    text: without_newline.to_owned(),
                }),
                anomaly: None,
            },
            (None, Some((stale_level, stale_text))) => {
                self.buffered = Some((level, text.to_owned()));
                DecodeResult {
                    output: DecodeOutput::Line(LogLine {
                        level: stale_level,
                        text: stale_text,
                    }),
                    anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
                }
            }
            (None, None) => {
                self.buffered = Some((level, text.to_owned()));
                DecodeResult {
                    output: DecodeOutput::None,
                    anomaly: None,
                }
            }
        }
    }
}

impl Default for LogDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::LogDecoder;
    use crate::decode_anomaly::DecodeAnomaly;
    use crate::decode_output::DecodeOutput;
    use crate::decode_result::DecodeResult;
    use crate::incoming_log_level::IncomingLogLevel;
    use crate::log_level::LogLevel;
    use crate::log_line::LogLine;

    #[test]
    fn feed_complete_info_line() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Info, "hello\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "hello".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_partial_without_newline() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Info, "hello");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::None,
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_cont_completion() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "hello ");
        let result = decoder.feed(IncomingLogLevel::Cont, "world\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "hello world".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_multi_part_cont() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "part1 ");
        decoder.feed(IncomingLogLevel::Cont, "part2 ");
        let result = decoder.feed(IncomingLogLevel::Cont, "part3\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "part1 part2 part3".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_non_cont_while_buffering() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "stale");
        let result = decoder.feed(IncomingLogLevel::Warn, "fresh\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::TwoLines {
                    earlier: LogLine {
                        level: LogLevel::Info,
                        text: "stale".to_owned(),
                    },
                    current: LogLine {
                        level: LogLevel::Warn,
                        text: "fresh".to_owned(),
                    },
                },
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            }
        );
    }

    #[test]
    fn feed_buffer_replacement() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "first");
        let result = decoder.feed(IncomingLogLevel::Warn, "second");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "first".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            }
        );

        let follow_up = decoder.feed(IncomingLogLevel::Cont, "more\n");
        assert_eq!(
            follow_up,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Warn,
                    text: "secondmore".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_orphan_cont() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Cont, "ghost\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::None,
                    text: "ghost".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        );
    }

    #[test]
    fn feed_orphan_cont_previous_level() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Warn, "complete\n");
        let result = decoder.feed(IncomingLogLevel::Cont, "ghost\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Warn,
                    text: "ghost".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        );
    }

    #[test]
    fn feed_none_level() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::None, "no-level\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::None,
                    text: "no-level".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn feed_unknown_level() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Unknown(9999), "weird\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Unknown(9999),
                    text: "weird".to_owned(),
                }),
                anomaly: None,
            }
        );
    }
}
