use crate::log_options::LogOptions;
use std::sync::OnceLock;
use tracing_core::{Interest, Kind, Metadata, callsite, field, identify_callsite};

static FIELD_NAMES: &[&str] = &["message", "module"];

struct OverridableFields {
    message: tracing::field::Field,
    target: tracing::field::Field,
}

macro_rules! log_cs {
    ($level:expr, $cs:ident, $meta:ident, $fields:ident, $ty:ident) => {
        struct $ty;
        static $cs: $ty = $ty;
        static $meta: Metadata<'static> = Metadata::new(
            "log event",
            "llama-cpp-bindings",
            $level,
            ::core::option::Option::None,
            ::core::option::Option::None,
            ::core::option::Option::None,
            field::FieldSet::new(FIELD_NAMES, identify_callsite!(&$cs)),
            Kind::EVENT,
        );
        static $fields: std::sync::LazyLock<OverridableFields> = std::sync::LazyLock::new(|| {
            let fields = $meta.fields();
            OverridableFields {
                message: fields
                    .field("message")
                    .expect("message field defined in FIELD_NAMES"),
                target: fields
                    .field("module")
                    .expect("module field defined in FIELD_NAMES"),
            }
        });

        impl callsite::Callsite for $ty {
            fn set_interest(&self, _: Interest) {}
            fn metadata(&self) -> &'static Metadata<'static> {
                &$meta
            }
        }
    };
}
log_cs!(
    tracing_core::Level::DEBUG,
    DEBUG_CS,
    DEBUG_META,
    DEBUG_FIELDS,
    DebugCallsite
);
log_cs!(
    tracing_core::Level::INFO,
    INFO_CS,
    INFO_META,
    INFO_FIELDS,
    InfoCallsite
);
log_cs!(
    tracing_core::Level::WARN,
    WARN_CS,
    WARN_META,
    WARN_FIELDS,
    WarnCallsite
);
log_cs!(
    tracing_core::Level::ERROR,
    ERROR_CS,
    ERROR_META,
    ERROR_FIELDS,
    ErrorCallsite
);

#[derive(Clone, Copy)]
pub enum Module {
    Ggml,
    LlamaCpp,
}

impl Module {
    const fn name(self) -> &'static str {
        match self {
            Self::Ggml => "ggml",
            Self::LlamaCpp => "llama.cpp",
        }
    }
}

fn meta_for_level(
    level: llama_cpp_bindings_sys::ggml_log_level,
) -> Option<(&'static Metadata<'static>, &'static OverridableFields)> {
    match level {
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG => Some((&DEBUG_META, &DEBUG_FIELDS)),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO => Some((&INFO_META, &INFO_FIELDS)),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN => Some((&WARN_META, &WARN_FIELDS)),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR => Some((&ERROR_META, &ERROR_FIELDS)),
        _ => None,
    }
}

pub struct State {
    pub options: LogOptions,
    module: Module,
    buffered: std::sync::Mutex<Option<(llama_cpp_bindings_sys::ggml_log_level, String)>>,
    previous_level: std::sync::atomic::AtomicI32,
    is_buffering: std::sync::atomic::AtomicBool,
}

impl State {
    #[must_use]
    pub fn new(module: Module, options: LogOptions) -> Self {
        Self {
            options,
            module,
            buffered: std::sync::Mutex::default(),
            previous_level: std::sync::atomic::AtomicI32::default(),
            is_buffering: std::sync::atomic::AtomicBool::default(),
        }
    }

    /// The match arms are duplicated per module because the `tracing` macros
    /// require the `target` argument to be a string literal — the upstream
    /// submodule name cannot be propagated dynamically.
    fn generate_log(&self, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        let (module, text) = text
            .char_indices()
            .take_while(|(_, ch)| ch.is_ascii_lowercase() || *ch == '_')
            .last()
            .and_then(|(pos, _)| {
                let next_two = text.get(pos + 1..pos + 3);
                if next_two == Some(": ") {
                    let (sub_module, text) = text.split_at(pos + 1);
                    let text = text.split_at(2).1;
                    Some((Some(format!("{}::{sub_module}", self.module.name())), text))
                } else {
                    None
                }
            })
            .unwrap_or((None, text));

        let effective_level = if self.options.demote_info_to_debug
            && (level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO
                || level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG)
        {
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG
        } else {
            level
        };

        let Some((meta, fields)) = meta_for_level(effective_level) else {
            tracing::warn!(
                level = effective_level,
                text = text,
                origin = "crate",
                "generate_log called with unmapped log level"
            );

            return;
        };

        tracing::dispatcher::get_default(|dispatcher| {
            dispatcher.event(&tracing::Event::new(
                meta,
                &meta.fields().value_set(&[
                    (&fields.message, Some(&text as &dyn tracing::field::Value)),
                    (
                        &fields.target,
                        module
                            .as_ref()
                            .map(|module_name| module_name as &dyn tracing::field::Value),
                    ),
                ]),
            ));
        });
    }

    /// Append more text to the previously buffered log.
    ///
    /// The text may or may not end with a newline.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn cont_buffered_log(&self, text: &str) {
        let mut lock = self.buffered.lock().unwrap();

        if let Some((previous_log_level, mut buffer)) = lock.take() {
            buffer.push_str(text);
            if buffer.ends_with('\n') {
                self.is_buffering
                    .store(false, std::sync::atomic::Ordering::Release);
                self.generate_log(previous_log_level, buffer.as_str());
            } else {
                *lock = Some((previous_log_level, buffer));
            }
        } else {
            let level = self
                .previous_level
                .load(std::sync::atomic::Ordering::Acquire)
                .cast_unsigned();
            tracing::warn!(
                inferred_level = level,
                text = text,
                origin = "crate",
                "llama.cpp sent out a CONT log without any previously buffered message"
            );
            *lock = Some((level, text.to_string()));
        }
    }

    /// Start buffering a message. Not the CONT log level and text is missing a newline.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn buffer_non_cont(&self, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        let replaced = self
            .buffered
            .lock()
            .unwrap()
            .replace((level, text.to_string()));

        if let Some((previous_log_level, buffer)) = replaced {
            tracing::warn!(
                level = previous_log_level,
                text = &buffer,
                origin = "crate",
                "Message buffered unnecessarily due to missing newline and not followed by a CONT"
            );
            self.generate_log(previous_log_level, buffer.as_str());
        }

        self.is_buffering
            .store(true, std::sync::atomic::Ordering::Release);
        self.previous_level
            .store(level.cast_signed(), std::sync::atomic::Ordering::Release);
    }

    /// Emit a normal unbuffered log message (not the CONT log level and the text ends with a newline).
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn emit_non_cont_line(&self, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        if self
            .is_buffering
            .swap(false, std::sync::atomic::Ordering::Acquire)
            && let Some((buf_level, buf_text)) = self.buffered.lock().unwrap().take()
        {
            tracing::warn!(
                level = buf_level,
                text = buf_text,
                origin = "crate",
                "llama.cpp message buffered spuriously due to missing \\n and being followed by a non-CONT message! (this indicates a bug within llama.cpp)"
            );
            self.generate_log(buf_level, buf_text.as_str());
        }

        self.previous_level
            .store(level.cast_signed(), std::sync::atomic::Ordering::Release);

        let (text, _trailing_newline) = text.split_at(text.len() - 1);

        match level {
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE => {
                if self.options.demote_info_to_debug {
                    self.generate_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG, text);
                } else {
                    tracing::info!(no_log_level = true, text);
                }
            }
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR => self.generate_log(level, text),
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT => {
                tracing::warn!(
                    text = text,
                    origin = "crate",
                    "CONT log level passed to emit_non_cont_line"
                );
            }
            _ => {
                tracing::warn!(
                    level = level,
                    text = text,
                    origin = "crate",
                    "Unknown llama.cpp log level"
                );
            }
        }
    }

    pub fn update_previous_level_for_disabled_log(
        &self,
        level: llama_cpp_bindings_sys::ggml_log_level,
    ) {
        if level != llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
            self.previous_level
                .store(level.cast_signed(), std::sync::atomic::Ordering::Release);
        }
    }

    /// Checks whether the given log level is enabled by the current tracing
    /// subscriber. CONT lines inherit the previous line's level rather than
    /// being checked on their own.
    pub fn is_enabled_for_level(&self, level: llama_cpp_bindings_sys::ggml_log_level) -> bool {
        let level = if level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
            self.previous_level
                .load(std::sync::atomic::Ordering::Relaxed)
                .cast_unsigned()
        } else {
            level
        };

        let effective_level = if self.options.demote_info_to_debug
            && (level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO
                || level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG)
        {
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG
        } else {
            level
        };

        let Some((meta, _)) = meta_for_level(effective_level) else {
            return false;
        };

        tracing::dispatcher::get_default(|dispatcher| dispatcher.enabled(meta))
    }
}

pub static LLAMA_STATE: OnceLock<Box<State>> = OnceLock::new();
pub static GGML_STATE: OnceLock<Box<State>> = OnceLock::new();

/// Bridges llama.cpp / ggml log callbacks into the `tracing` ecosystem.
///
/// The fast path — newline-terminated DEBUG/INFO/WARN/ERROR lines — must avoid
/// taking the log state lock and must not allocate, so the buffering and
/// CONT-handling logic only runs on the slow path. Lines that lack a trailing
/// newline are buffered: their absence is the only signal upstream uses to
/// announce that a CONT message will follow, and we cannot distinguish that
/// from a typo until the next message arrives.
extern "C" fn logs_to_trace(
    level: llama_cpp_bindings_sys::ggml_log_level,
    text: *const ::std::os::raw::c_char,
    data: *mut ::std::os::raw::c_void,
) {
    use std::borrow::Borrow;

    let log_state = unsafe { &*(data as *const State) };

    if log_state.options.disabled {
        return;
    }

    if !log_state.is_enabled_for_level(level) {
        log_state.update_previous_level_for_disabled_log(level);

        return;
    }

    let text = unsafe { std::ffi::CStr::from_ptr(text) };
    let text = text.to_string_lossy();
    let text: &str = text.borrow();

    if level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
        log_state.cont_buffered_log(text);
    } else if text.ends_with('\n') {
        log_state.emit_non_cont_line(level, text);
    } else {
        log_state.buffer_non_cont(level, text);
    }
}

/// Redirect llama.cpp logs into tracing.
///
/// `llama.cpp` and `ggml` are wired up to separate `State` instances so a CONT
/// line emitted by one cannot be appended to a buffered line from the other.
/// `llama_log_set` also installs the callback for `ggml`, so the `ggml_log_set`
/// call must come second to override that and bind the ggml state explicitly.
pub fn send_logs_to_tracing(options: LogOptions) {
    let llama_heap_state = Box::as_ref(
        LLAMA_STATE.get_or_init(|| Box::new(State::new(Module::LlamaCpp, options.clone()))),
    ) as *const _;
    let ggml_heap_state =
        Box::as_ref(GGML_STATE.get_or_init(|| Box::new(State::new(Module::Ggml, options))))
            as *const _;

    unsafe {
        llama_cpp_bindings_sys::llama_log_set(Some(logs_to_trace), llama_heap_state as *mut _);
        llama_cpp_bindings_sys::ggml_log_set(Some(logs_to_trace), ggml_heap_state as *mut _);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::util::SubscriberInitExt;

    use super::{Module, State, logs_to_trace};
    use crate::log_options::LogOptions;

    #[test]
    fn module_name_ggml() {
        assert_eq!(Module::Ggml.name(), "ggml");
    }

    #[test]
    fn module_name_llama_cpp() {
        assert_eq!(Module::LlamaCpp.name(), "llama.cpp");
    }

    #[test]
    fn state_new_creates_empty_buffer() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());
        let buffer = state
            .buffered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        assert!(buffer.is_none());
        drop(buffer);
        assert!(!state.options.disabled);
    }

    #[test]
    fn update_previous_level_for_disabled_log_stores_level() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN);

        let stored = state
            .previous_level
            .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(
            stored,
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN.cast_signed()
        );
    }

    #[test]
    fn update_previous_level_ignores_cont() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR);
        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT);

        let stored = state
            .previous_level
            .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(
            stored,
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR.cast_signed()
        );
    }

    #[test]
    fn buffer_non_cont_sets_buffering_flag() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "partial");

        assert!(
            state
                .is_buffering
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        let buffer = state
            .buffered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        assert!(buffer.is_some());
        let (level, text) = buffer.as_ref().unwrap();
        assert_eq!(*level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO);
        assert_eq!(text, "partial");
        drop(buffer);
    }

    #[test]
    fn cont_buffered_log_appends_to_existing_buffer() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "hello ");

        state.cont_buffered_log("world");

        let buffer = state
            .buffered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        assert!(buffer.is_some());
        let (_, text) = buffer.as_ref().unwrap();
        assert_eq!(text, "hello world");
        drop(buffer);
    }

    struct Logger {
        #[expect(
            unused,
            reason = "guard must outlive the test body so the tracing subscriber stays installed; \
                      dropping it un-installs the subscriber and tests would silently miss log lines"
        )]
        guard: tracing::subscriber::DefaultGuard,
        logs: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<String>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let log_line = String::from_utf8_lossy(buf).into_owned();
            self.0.lock().unwrap().push(log_line);

            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn create_logger(max_level: tracing::Level) -> Logger {
        let logs = Arc::new(Mutex::new(vec![]));
        let writer = VecWriter(logs.clone());

        Logger {
            guard: tracing_subscriber::fmt()
                .with_max_level(max_level)
                .with_ansi(false)
                .without_time()
                .with_file(false)
                .with_line_number(false)
                .with_level(false)
                .with_target(false)
                .with_writer(move || writer.clone())
                .finish()
                .set_default(),
            logs,
        }
    }

    #[test]
    fn cont_disabled_log() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"world\n".as_ptr(),
            log_ptr,
        );

        assert!(logger.logs.lock().unwrap().is_empty());

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"world".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"\n".as_ptr(),
            log_ptr,
        );
    }

    #[test]
    fn cont_message_concatenates_payload_then_flush_appends_extra_newline() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        let cont_payload_with_newline = c"world\n";
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            cont_payload_with_newline.as_ptr(),
            log_ptr,
        );

        let payload_newline = '\n';
        let flush_appended_newline = '\n';
        assert_eq!(
            *logger.logs.lock().unwrap(),
            vec![format!(
                "Hello world{payload_newline}{flush_appended_newline}"
            )]
        );
    }

    #[test]
    fn disabled_logs_are_suppressed() {
        let logger = create_logger(tracing::Level::DEBUG);
        let disabled_options = LogOptions::default().with_logs_enabled(false);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, disabled_options));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"Should not appear\n".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR,
            c"Also suppressed\n".as_ptr(),
            log_ptr,
        );

        assert!(logger.logs.lock().unwrap().is_empty());
    }

    #[test]
    fn info_level_log_emitted() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"info message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("info message"));
        drop(logs);
    }

    #[test]
    fn warn_level_log_emitted() {
        let logger = create_logger(tracing::Level::WARN);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN,
            c"warning message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("warning message"));
        drop(logs);
    }

    #[test]
    fn error_level_log_emitted() {
        let logger = create_logger(tracing::Level::ERROR);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR,
            c"error message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("error message"));
        drop(logs);
    }

    #[test]
    fn debug_level_log_emitted_when_enabled() {
        let logger = create_logger(tracing::Level::DEBUG);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"debug message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("debug message"));
        drop(logs);
    }

    #[test]
    fn submodule_extraction_from_log_text() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"sampling: initialized\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("initialized"));
        drop(logs);
    }

    #[test]
    fn multi_part_cont_log() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"part1 ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"part2 ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"part3\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("part1 part2 part3"));
        drop(logs);
    }

    #[test]
    fn demote_info_to_debug_suppresses_info_under_info_subscriber() {
        let logger = create_logger(tracing::Level::INFO);
        let options = LogOptions::default().with_demote_info_to_debug(true);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, options));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"should be suppressed\n".as_ptr(),
            log_ptr,
        );

        assert!(logger.logs.lock().unwrap().is_empty());
    }

    #[test]
    fn demote_info_to_debug_emits_info_under_debug_subscriber() {
        let logger = create_logger(tracing::Level::DEBUG);
        let options = LogOptions::default().with_demote_info_to_debug(true);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, options));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"visible at debug\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("visible at debug"));
        drop(logs);
    }

    #[test]
    fn demote_info_to_debug_preserves_error_under_info_subscriber() {
        let logger = create_logger(tracing::Level::INFO);
        let options = LogOptions::default().with_demote_info_to_debug(true);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, options));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR,
            c"error still visible\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("error still visible"));
        drop(logs);
    }

    #[test]
    fn demote_info_to_debug_preserves_warn_under_info_subscriber() {
        let logger = create_logger(tracing::Level::INFO);
        let options = LogOptions::default().with_demote_info_to_debug(true);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, options));
        let log_ptr =
            std::ptr::from_mut::<State>(log_state.as_mut()).cast::<std::os::raw::c_void>();

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN,
            c"warning still visible\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("warning still visible"));
        drop(logs);
    }

    #[test]
    fn emit_non_cont_line_level_none() {
        let logger = create_logger(tracing::Level::INFO);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.emit_non_cont_line(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
            "none level message\n",
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("none level message"));
        drop(logs);
    }

    #[test]
    fn emit_non_cont_line_level_none_demoted_to_debug() {
        let logger = create_logger(tracing::Level::DEBUG);
        let options = LogOptions::default().with_demote_info_to_debug(true);
        let state = State::new(Module::LlamaCpp, options);

        state.emit_non_cont_line(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
            "demoted none\n",
        );

        let logs = logger.logs.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("demoted none"));
        drop(logs);
    }

    #[test]
    fn cont_without_prior_buffer_infers_level() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN);
        state.cont_buffered_log("orphan text");

        let buffer = state.buffered.lock().unwrap();
        assert!(buffer.is_some());
        let (level, text) = buffer.as_ref().unwrap();
        assert_eq!(*level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN);
        assert_eq!(text, "orphan text");
        drop(buffer);
    }

    #[test]
    fn emit_non_cont_flushes_stale_buffer() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "stale");

        state.emit_non_cont_line(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN, "new line\n");

        let buffer = state.buffered.lock().unwrap();
        assert!(buffer.is_none());
        drop(buffer);
    }

    #[test]
    fn buffer_non_cont_replaces_previous_buffer() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "first");
        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN, "second");

        let buffer = state.buffered.lock().unwrap();
        let (level, text) = buffer.as_ref().unwrap();
        assert_eq!(*level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN);
        assert_eq!(text, "second");
        drop(buffer);
    }

    #[test]
    fn is_enabled_for_cont_uses_previous_level() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR);

        let enabled = state.is_enabled_for_level(llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT);

        assert!(enabled);
    }

    #[test]
    fn unknown_log_level_emits_warning() {
        let logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.emit_non_cont_line(9999, "unknown level message\n");

        let logs = logger.logs.lock().unwrap();
        assert!(
            logs.iter()
                .any(|log_line| log_line.contains("Unknown llama.cpp log level"))
        );
        drop(logs);
    }

    #[test]
    fn send_logs_to_tracing_initializes_global_states() {
        use super::{GGML_STATE, LLAMA_STATE, send_logs_to_tracing};

        send_logs_to_tracing(LogOptions::default());

        assert!(LLAMA_STATE.get().is_some());
        assert!(GGML_STATE.get().is_some());
    }

    #[test]
    fn meta_for_level_returns_none_for_unknown_level() {
        let result = super::meta_for_level(9999);

        assert!(result.is_none());
    }

    #[test]
    fn is_enabled_for_level_returns_false_for_none_level() {
        let _logger = create_logger(tracing::Level::DEBUG);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        let enabled = state.is_enabled_for_level(llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE);

        assert!(!enabled);
    }

    #[test]
    fn generate_log_handles_unmapped_level_gracefully() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.generate_log(9999, "unmapped level message");
    }

    #[test]
    fn emit_non_cont_line_handles_cont_level_gracefully() {
        let _logger = create_logger(tracing::Level::WARN);
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.emit_non_cont_line(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            "cont passed to non-cont\n",
        );
    }

    #[test]
    fn callsite_metadata_returns_static_metadata() {
        use tracing_core::callsite::Callsite;

        let debug_meta = super::DEBUG_CS.metadata();
        let info_meta = super::INFO_CS.metadata();
        let warn_meta = super::WARN_CS.metadata();
        let error_meta = super::ERROR_CS.metadata();

        assert_eq!(*debug_meta.level(), tracing_core::Level::DEBUG);
        assert_eq!(*info_meta.level(), tracing_core::Level::INFO);
        assert_eq!(*warn_meta.level(), tracing_core::Level::WARN);
        assert_eq!(*error_meta.level(), tracing_core::Level::ERROR);
    }

    #[test]
    fn callsite_set_interest_does_not_panic() {
        use tracing_core::callsite::Callsite;
        use tracing_core::subscriber::Interest;

        super::DEBUG_CS.set_interest(Interest::always());
        super::INFO_CS.set_interest(Interest::never());
        super::WARN_CS.set_interest(Interest::sometimes());
        super::ERROR_CS.set_interest(Interest::always());
    }

    #[test]
    fn vec_writer_flush_succeeds() {
        use std::io::Write;

        let mut writer = VecWriter(Arc::new(Mutex::new(vec![])));

        writer.flush().unwrap();
    }
}
