#![deny(clippy::expect_used)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::panic)]
#![deny(clippy::unwrap_used)]

use std::sync::{Mutex, OnceLock};

use llama_cpp_log_decoder::decode_anomaly::DecodeAnomaly;
use llama_cpp_log_decoder::decode_output::DecodeOutput;
use llama_cpp_log_decoder::incoming_log_level::IncomingLogLevel;
use llama_cpp_log_decoder::log_decoder::LogDecoder;
use llama_cpp_log_decoder::log_level::LogLevel;
use llama_cpp_log_decoder::log_line::LogLine;

use crate::log_options::LogOptions;

struct LogSource {
    decoder: Mutex<LogDecoder>,
    target: &'static str,
    options: LogOptions,
}

impl LogSource {
    const fn new(target: &'static str, options: LogOptions) -> Self {
        Self {
            decoder: Mutex::new(LogDecoder::new()),
            target,
            options,
        }
    }
}

static LLAMA_SOURCE: OnceLock<LogSource> = OnceLock::new();
static GGML_SOURCE: OnceLock<LogSource> = OnceLock::new();

#[cfg(target_env = "msvc")]
const fn ggml_level_to_u32(level: llama_cpp_bindings_sys::ggml_log_level) -> u32 {
    level.cast_unsigned()
}

#[cfg(not(target_env = "msvc"))]
const fn ggml_level_to_u32(level: llama_cpp_bindings_sys::ggml_log_level) -> u32 {
    level
}

const fn ggml_level_to_incoming(raw: llama_cpp_bindings_sys::ggml_log_level) -> IncomingLogLevel {
    match raw {
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE => IncomingLogLevel::None,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG => IncomingLogLevel::Debug,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO => IncomingLogLevel::Info,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN => IncomingLogLevel::Warn,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR => IncomingLogLevel::Error,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT => IncomingLogLevel::Cont,
        other => IncomingLogLevel::Unknown(ggml_level_to_u32(other)),
    }
}

fn resolve_record(line: LogLine, demote_info_to_debug: bool) -> (log::Level, String) {
    let effective_level =
        if demote_info_to_debug && matches!(line.level, LogLevel::Info | LogLevel::None) {
            LogLevel::Debug
        } else {
            line.level
        };

    match effective_level {
        LogLevel::Debug => (log::Level::Debug, line.text),
        LogLevel::Info | LogLevel::None => (log::Level::Info, line.text),
        LogLevel::Warn => (log::Level::Warn, line.text),
        LogLevel::Error => (log::Level::Error, line.text),
        LogLevel::Unknown(raw) => (
            log::Level::Warn,
            format!("[unknown level {raw}] {}", line.text),
        ),
    }
}

fn dispatch_line(source: &LogSource, line: LogLine) {
    let (level, message) = resolve_record(line, source.options.demote_info_to_debug);
    log::log!(target: source.target, level, "{message}");
}

fn dispatch_output(source: &LogSource, output: DecodeOutput) {
    match output {
        DecodeOutput::None => {}
        DecodeOutput::Line(line) => dispatch_line(source, line),
        DecodeOutput::TwoLines { earlier, current } => {
            dispatch_line(source, earlier);
            dispatch_line(source, current);
        }
    }
}

fn dispatch_anomaly(source: &LogSource, anomaly: DecodeAnomaly) {
    log::warn!(
        target: source.target,
        "llama.cpp log decoder anomaly: {anomaly:?}",
    );
}

unsafe extern "C" fn logs_to_log(
    raw_level: llama_cpp_bindings_sys::ggml_log_level,
    text_ptr: *const std::os::raw::c_char,
    data_ptr: *mut std::os::raw::c_void,
) {
    let source: &LogSource = unsafe { &*data_ptr.cast::<LogSource>() };

    if source.options.disabled {
        return;
    }

    if text_ptr.is_null() {
        log::warn!(
            target: source.target,
            "received NULL text pointer from llama.cpp log callback",
        );
        return;
    }

    let text_cstr = unsafe { std::ffi::CStr::from_ptr(text_ptr) };
    let text = text_cstr.to_string_lossy();

    let incoming = ggml_level_to_incoming(raw_level);

    let result = {
        let mut decoder = source
            .decoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        decoder.feed(incoming, &text)
    };

    dispatch_output(source, result.output);

    if let Some(anomaly) = result.anomaly {
        dispatch_anomaly(source, anomaly);
    }
}

pub fn send_logs_to_log(options: LogOptions) {
    let llama_source: *const LogSource =
        LLAMA_SOURCE.get_or_init(|| LogSource::new("llama.cpp", options.clone()));
    let ggml_source: *const LogSource = GGML_SOURCE.get_or_init(|| LogSource::new("ggml", options));

    unsafe {
        llama_cpp_bindings_sys::llama_log_set(
            Some(logs_to_log),
            llama_source.cast::<std::os::raw::c_void>().cast_mut(),
        );
        llama_cpp_bindings_sys::ggml_log_set(
            Some(logs_to_log),
            ggml_source.cast::<std::os::raw::c_void>().cast_mut(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, Once};

    use llama_cpp_log_decoder::decode_output::DecodeOutput;
    use llama_cpp_log_decoder::incoming_log_level::IncomingLogLevel;
    use llama_cpp_log_decoder::log_level::LogLevel;
    use llama_cpp_log_decoder::log_line::LogLine;
    use log::{Level, Log, Metadata, Record};
    use serial_test::serial;

    use super::{
        GGML_SOURCE, LLAMA_SOURCE, LogSource, dispatch_output, ggml_level_to_incoming, logs_to_log,
        resolve_record, send_logs_to_log,
    };
    use crate::log_options::LogOptions;

    #[derive(Clone, Debug)]
    struct CapturedRecord {
        level: Level,
        target: String,
        message: String,
    }

    struct TestLogger {
        records: Mutex<Vec<CapturedRecord>>,
    }

    impl Log for TestLogger {
        fn enabled(&self, _: &Metadata) -> bool {
            true
        }

        fn log(&self, record: &Record) {
            let mut guard = self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.push(CapturedRecord {
                level: record.level(),
                target: record.target().to_owned(),
                message: record.args().to_string(),
            });
        }

        fn flush(&self) {}
    }

    static TEST_LOGGER: TestLogger = TestLogger {
        records: Mutex::new(Vec::new()),
    };
    static INSTALL: Once = Once::new();

    fn ensure_test_logger_installed() {
        INSTALL.call_once(|| {
            if log::set_logger(&TEST_LOGGER).is_ok() {
                log::set_max_level(log::LevelFilter::Trace);
            }
        });
    }

    fn records_for(target: &str) -> Vec<CapturedRecord> {
        let guard = TEST_LOGGER
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .iter()
            .filter(|record| record.target == target)
            .cloned()
            .collect()
    }

    fn invoke_callback(
        level: llama_cpp_bindings_sys::ggml_log_level,
        text: &std::ffi::CStr,
        source: &LogSource,
    ) {
        let ptr = std::ptr::from_ref(source)
            .cast::<std::os::raw::c_void>()
            .cast_mut();
        unsafe {
            logs_to_log(level, text.as_ptr(), ptr);
        }
    }

    #[test]
    fn test_logger_enabled_and_flush() {
        let metadata = Metadata::builder()
            .level(Level::Info)
            .target("test-logger-enabled")
            .build();

        assert!(TEST_LOGGER.enabled(&metadata));
        TEST_LOGGER.flush();
    }

    #[test]
    fn ggml_level_to_incoming_known_constants() {
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE),
            IncomingLogLevel::None,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG),
            IncomingLogLevel::Debug,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO),
            IncomingLogLevel::Info,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN),
            IncomingLogLevel::Warn,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR),
            IncomingLogLevel::Error,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT),
            IncomingLogLevel::Cont,
        );
    }

    #[test]
    fn ggml_level_to_incoming_unknown_value() {
        assert_eq!(
            ggml_level_to_incoming(9999),
            IncomingLogLevel::Unknown(9999)
        );
    }

    #[test]
    fn dispatch_when_disabled() {
        ensure_test_logger_installed();

        let target = "test-dispatch-when-disabled";
        let source = LogSource::new(target, LogOptions::default().with_logs_enabled(false));
        invoke_callback(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"hello\n",
            &source,
        );

        assert!(records_for(target).is_empty());
    }

    #[test]
    fn demote_info_to_debug_on_info() {
        ensure_test_logger_installed();

        let target = "test-demote-info-on-info";
        let source = LogSource::new(
            target,
            LogOptions::default().with_demote_info_to_debug(true),
        );
        invoke_callback(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"info-line\n",
            &source,
        );

        assert!(records_for(target).iter().any(|record| {
            record.level == Level::Debug && record.message.contains("info-line")
        }));
    }

    #[test]
    fn demote_info_to_debug_on_warn() {
        ensure_test_logger_installed();

        let target = "test-demote-info-on-warn";
        let source = LogSource::new(
            target,
            LogOptions::default().with_demote_info_to_debug(true),
        );
        invoke_callback(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN,
            c"warn-line\n",
            &source,
        );

        assert!(
            records_for(target).iter().any(|record| {
                record.level == Level::Warn && record.message.contains("warn-line")
            })
        );
    }

    #[test]
    fn dispatch_unknown_level() {
        ensure_test_logger_installed();

        let target = "test-dispatch-unknown-level";
        let source = LogSource::new(target, LogOptions::default());
        invoke_callback(9999, c"weird\n", &source);

        assert!(records_for(target).iter().any(|record| {
            record.level == Level::Warn
                && record.message.contains("[unknown level 9999]")
                && record.message.contains("weird")
        }));
    }

    #[test]
    fn dispatch_orphan_cont_anomaly() {
        ensure_test_logger_installed();

        let target = "test-dispatch-orphan-cont";
        let source = LogSource::new(target, LogOptions::default());
        invoke_callback(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"ghost\n",
            &source,
        );

        assert!(records_for(target).iter().any(|record| {
            record.level == Level::Warn && record.message.contains("OrphanCont")
        }));
    }

    #[test]
    fn resolve_record_error_level_maps_to_error_level() {
        let (level, message) = resolve_record(
            LogLine {
                level: LogLevel::Error,
                text: "boom".to_owned(),
            },
            false,
        );

        assert_eq!(level, Level::Error);
        assert_eq!(message, "boom");
    }

    #[test]
    fn dispatch_output_none_emits_no_records() {
        ensure_test_logger_installed();

        let target = "test-dispatch-output-none";
        let source = LogSource::new(target, LogOptions::default());
        dispatch_output(&source, DecodeOutput::None);

        assert!(records_for(target).is_empty());
    }

    #[test]
    fn dispatch_output_two_lines_emits_both_records() {
        ensure_test_logger_installed();

        let target = "test-dispatch-output-two-lines";
        let source = LogSource::new(target, LogOptions::default());
        dispatch_output(
            &source,
            DecodeOutput::TwoLines {
                earlier: LogLine {
                    level: LogLevel::Info,
                    text: "earlier-line".to_owned(),
                },
                current: LogLine {
                    level: LogLevel::Warn,
                    text: "current-line".to_owned(),
                },
            },
        );

        let records = records_for(target);
        assert!(
            records
                .iter()
                .any(|record| record.message.contains("earlier-line"))
        );
        assert!(
            records
                .iter()
                .any(|record| record.message.contains("current-line"))
        );
    }

    #[test]
    #[serial]
    fn send_logs_to_log_initialization() {
        ensure_test_logger_installed();
        send_logs_to_log(LogOptions::default());

        assert!(LLAMA_SOURCE.get().is_some());
        assert!(GGML_SOURCE.get().is_some());
    }

    #[test]
    fn null_text_pointer() {
        ensure_test_logger_installed();

        let target = "test-null-text-pointer";
        let source = LogSource::new(target, LogOptions::default());
        let source_ptr = std::ptr::from_ref(&source)
            .cast::<std::os::raw::c_void>()
            .cast_mut();
        unsafe {
            logs_to_log(
                llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
                std::ptr::null(),
                source_ptr,
            );
        }

        assert!(records_for(target).iter().any(|record| {
            record.level == Level::Warn && record.message.contains("NULL text pointer")
        }));
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "deliberate panic to poison the decoder mutex for fault-injection coverage"
    )]
    fn decoder_mutex_poison() {
        ensure_test_logger_installed();

        let target = "test-decoder-mutex-poison";
        let source = LogSource::new(target, LogOptions::default());

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let _guard = source
                    .decoder
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                panic!("intentional poison");
            });
            let _ = handle.join();
        });

        assert!(source.decoder.is_poisoned());

        invoke_callback(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"after-poison\n",
            &source,
        );

        assert!(
            records_for(target)
                .iter()
                .any(|record| record.message.contains("after-poison"))
        );
    }
}
