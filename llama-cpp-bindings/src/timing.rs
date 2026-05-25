use std::fmt::{Debug, Display, Formatter};

#[derive(Clone, Copy, Debug)]
pub struct LlamaTimings {
    pub timings: llama_cpp_bindings_sys::llama_perf_context_data,
}

impl LlamaTimings {
    #[must_use]
    pub const fn new(
        t_start_ms: f64,
        t_load_ms: f64,
        t_p_eval_ms: f64,
        t_eval_ms: f64,
        n_p_eval: i32,
        n_eval: i32,
        n_reused: i32,
    ) -> Self {
        Self {
            timings: llama_cpp_bindings_sys::llama_perf_context_data {
                t_start_ms,
                t_load_ms,
                t_p_eval_ms,
                t_eval_ms,
                n_p_eval,
                n_eval,
                n_reused,
            },
        }
    }

    #[must_use]
    pub const fn t_start_ms(&self) -> f64 {
        self.timings.t_start_ms
    }

    #[must_use]
    pub const fn t_load_ms(&self) -> f64 {
        self.timings.t_load_ms
    }

    #[must_use]
    pub const fn t_p_eval_ms(&self) -> f64 {
        self.timings.t_p_eval_ms
    }

    #[must_use]
    pub const fn t_eval_ms(&self) -> f64 {
        self.timings.t_eval_ms
    }

    #[must_use]
    pub const fn n_p_eval(&self) -> i32 {
        self.timings.n_p_eval
    }

    #[must_use]
    pub const fn n_eval(&self) -> i32 {
        self.timings.n_eval
    }

    pub const fn set_t_start_ms(&mut self, t_start_ms: f64) {
        self.timings.t_start_ms = t_start_ms;
    }

    pub const fn set_t_load_ms(&mut self, t_load_ms: f64) {
        self.timings.t_load_ms = t_load_ms;
    }

    pub const fn set_t_p_eval_ms(&mut self, t_p_eval_ms: f64) {
        self.timings.t_p_eval_ms = t_p_eval_ms;
    }

    pub const fn set_t_eval_ms(&mut self, t_eval_ms: f64) {
        self.timings.t_eval_ms = t_eval_ms;
    }

    pub const fn set_n_p_eval(&mut self, n_p_eval: i32) {
        self.timings.n_p_eval = n_p_eval;
    }

    pub const fn set_n_eval(&mut self, n_eval: i32) {
        self.timings.n_eval = n_eval;
    }
}

fn write_timings(timings: &LlamaTimings, writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
    writeln!(writer, "load time = {:.2} ms", timings.t_load_ms())?;

    if timings.n_p_eval() > 0 {
        writeln!(
            writer,
            "prompt eval time = {:.2} ms / {} tokens ({:.2} ms per token, {:.2} tokens per second)",
            timings.t_p_eval_ms(),
            timings.n_p_eval(),
            timings.t_p_eval_ms() / f64::from(timings.n_p_eval()),
            1e3 / timings.t_p_eval_ms() * f64::from(timings.n_p_eval())
        )?;
    } else {
        writeln!(
            writer,
            "prompt eval time = {:.2} ms / 0 tokens",
            timings.t_p_eval_ms(),
        )?;
    }

    if timings.n_eval() > 0 {
        writeln!(
            writer,
            "eval time = {:.2} ms / {} runs ({:.2} ms per token, {:.2} tokens per second)",
            timings.t_eval_ms(),
            timings.n_eval(),
            timings.t_eval_ms() / f64::from(timings.n_eval()),
            1e3 / timings.t_eval_ms() * f64::from(timings.n_eval())
        )?;
    } else {
        writeln!(writer, "eval time = {:.2} ms / 0 runs", timings.t_eval_ms())?;
    }

    Ok(())
}

impl Display for LlamaTimings {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write_timings(self, formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaTimings;

    #[test]
    fn display_format_with_valid_counts() {
        let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);
        let output = format!("{timings}");

        assert!(output.contains("load time = 2.00 ms"));
        assert!(output.contains("prompt eval time = 3.00 ms / 5 tokens"));
        assert!(output.contains("eval time = 4.00 ms / 6 runs"));
    }

    #[test]
    fn display_format_handles_zero_eval_counts() {
        let timings = LlamaTimings::new(0.0, 1.0, 2.0, 3.0, 0, 0, 0);
        let output = format!("{timings}");

        assert!(output.contains("load time = 1.00 ms"));
        assert!(output.contains("prompt eval time = 2.00 ms / 0 tokens"));
        assert!(output.contains("eval time = 3.00 ms / 0 runs"));
        assert!(!output.contains("NaN"));
        assert!(!output.contains("inf"));
    }

    #[test]
    fn set_t_start_ms() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_t_start_ms(42.0);

        assert!((timings.t_start_ms() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn set_t_load_ms() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_t_load_ms(10.5);

        assert!((timings.t_load_ms() - 10.5).abs() < f64::EPSILON);
    }

    #[test]
    fn set_t_p_eval_ms() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_t_p_eval_ms(7.7);

        assert!((timings.t_p_eval_ms() - 7.7).abs() < f64::EPSILON);
    }

    #[test]
    fn set_t_eval_ms() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_t_eval_ms(3.3);

        assert!((timings.t_eval_ms() - 3.3).abs() < f64::EPSILON);
    }

    #[test]
    fn set_n_p_eval() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_n_p_eval(100);

        assert_eq!(timings.n_p_eval(), 100);
    }

    #[test]
    fn set_n_eval() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_n_eval(200);

        assert_eq!(timings.n_eval(), 200);
    }

    #[test]
    fn write_timings_propagates_writer_errors() {
        struct FailingWriter;

        impl std::fmt::Write for FailingWriter {
            fn write_str(&mut self, _text: &str) -> std::fmt::Result {
                Err(std::fmt::Error)
            }
        }

        let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);
        let result = super::write_timings(&timings, &mut FailingWriter);

        assert!(result.is_err());
    }

    #[test]
    fn write_timings_zero_p_eval_with_failing_writer() {
        struct FailAfterNWrites {
            remaining: usize,
        }

        impl std::fmt::Write for FailAfterNWrites {
            fn write_str(&mut self, _text: &str) -> std::fmt::Result {
                if self.remaining == 0 {
                    return Err(std::fmt::Error);
                }
                self.remaining -= 1;

                Ok(())
            }
        }

        let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 0, 6, 1);
        let result = super::write_timings(&timings, &mut FailAfterNWrites { remaining: 1 });

        assert!(result.is_err());
    }

    #[test]
    fn write_timings_fails_at_each_writeln_boundary() {
        struct FailAfterNWrites {
            remaining: usize,
        }

        impl std::fmt::Write for FailAfterNWrites {
            fn write_str(&mut self, _text: &str) -> std::fmt::Result {
                if self.remaining == 0 {
                    return Err(std::fmt::Error);
                }
                self.remaining -= 1;

                Ok(())
            }
        }

        let with_counts = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);
        let zero_counts = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 0, 0, 1);

        for writes_allowed in 0..20 {
            let _ = super::write_timings(
                &with_counts,
                &mut FailAfterNWrites {
                    remaining: writes_allowed,
                },
            );
            let _ = super::write_timings(
                &zero_counts,
                &mut FailAfterNWrites {
                    remaining: writes_allowed,
                },
            );
        }
    }
}
