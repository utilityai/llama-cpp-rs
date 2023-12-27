//! Safe wrapper around `llama_timings`.
use std::fmt::{Debug, Display, Formatter};

/// A wrapper around `llama_timings`.
#[derive(Clone, Copy, Debug)]
pub struct LlamaTimings {
    pub(crate) timings: llama_cpp_sys::llama_timings,
}

impl LlamaTimings {
    /// Create a new `LlamaTimings`.
    /// ```
    /// # use llama_cpp::timing::LlamaTimings;
    /// let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9);
    /// let timings_str = "load time = 3.00 ms
    /// sample time = 4.00 ms / 7 runs (0.57 ms per token, 1750.00 tokens per second)
    /// prompt eval time = 5.00 ms / 8 tokens (0.62 ms per token, 1600.00 tokens per second)
    /// eval time = 6.00 ms / 9 runs (0.67 ms per token, 1500.00 tokens per second)
    /// total time = 1.00 ms";
    /// assert_eq!(timings_str, format!("{}", timings));
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        t_start_ms: f64,
        t_end_ms: f64,
        t_load_ms: f64,
        t_sample_ms: f64,
        t_p_eval_ms: f64,
        t_eval_ms: f64,
        n_sample: i32,
        n_p_eval: i32,
        n_eval: i32,
    ) -> Self {
        Self {
            timings: llama_cpp_sys::llama_timings {
                t_start_ms,
                t_end_ms,
                t_load_ms,
                t_sample_ms,
                t_p_eval_ms,
                t_eval_ms,
                n_sample,
                n_p_eval,
                n_eval,
            },
        }
    }

    /// Get the start time in milliseconds.
    #[must_use]
    pub fn t_start_ms(&self) -> f64 {
        self.timings.t_start_ms
    }

    /// Get the end time in milliseconds.
    #[must_use]
    pub fn t_end_ms(&self) -> f64 {
        self.timings.t_end_ms
    }

    /// Get the load time in milliseconds.
    #[must_use]
    pub fn t_load_ms(&self) -> f64 {
        self.timings.t_load_ms
    }

    /// Get the sample time in milliseconds.
    #[must_use]
    pub fn t_sample_ms(&self) -> f64 {
        self.timings.t_sample_ms
    }

    /// Get the prompt evaluation time in milliseconds.
    #[must_use]
    pub fn t_p_eval_ms(&self) -> f64 {
        self.timings.t_p_eval_ms
    }

    /// Get the evaluation time in milliseconds.
    #[must_use]
    pub fn t_eval_ms(&self) -> f64 {
        self.timings.t_eval_ms
    }

    /// Get the number of samples.
    #[must_use]
    pub fn n_sample(&self) -> i32 {
        self.timings.n_sample
    }

    /// Get the number of prompt evaluations.
    #[must_use]
    pub fn n_p_eval(&self) -> i32 {
        self.timings.n_p_eval
    }

    /// Get the number of evaluations.
    #[must_use]
    pub fn n_eval(&self) -> i32 {
        self.timings.n_eval
    }

    /// Set the start time in milliseconds.
    pub fn set_t_start_ms(&mut self, t_start_ms: f64) {
        self.timings.t_start_ms = t_start_ms;
    }

    /// Set the end time in milliseconds.
    pub fn set_t_end_ms(&mut self, t_end_ms: f64) {
        self.timings.t_end_ms = t_end_ms;
    }

    /// Set the load time in milliseconds.
    pub fn set_t_load_ms(&mut self, t_load_ms: f64) {
        self.timings.t_load_ms = t_load_ms;
    }

    /// Set the sample time in milliseconds.
    pub fn set_t_sample_ms(&mut self, t_sample_ms: f64) {
        self.timings.t_sample_ms = t_sample_ms;
    }

    /// Set the prompt evaluation time in milliseconds.
    pub fn set_t_p_eval_ms(&mut self, t_p_eval_ms: f64) {
        self.timings.t_p_eval_ms = t_p_eval_ms;
    }

    /// Set the evaluation time in milliseconds.
    pub fn set_t_eval_ms(&mut self, t_eval_ms: f64) {
        self.timings.t_eval_ms = t_eval_ms;
    }

    /// Set the number of samples.
    pub fn set_n_sample(&mut self, n_sample: i32) {
        self.timings.n_sample = n_sample;
    }

    /// Set the number of prompt evaluations.
    pub fn set_n_p_eval(&mut self, n_p_eval: i32) {
        self.timings.n_p_eval = n_p_eval;
    }

    /// Set the number of evaluations.
    pub fn set_n_eval(&mut self, n_eval: i32) {
        self.timings.n_eval = n_eval;
    }
}

impl Display for LlamaTimings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "load time = {:.2} ms", self.t_load_ms())?;
        writeln!(
            f,
            "sample time = {:.2} ms / {} runs ({:.2} ms per token, {:.2} tokens per second)",
            self.t_sample_ms(),
            self.n_sample(),
            self.t_sample_ms() / f64::from(self.n_sample()),
            1e3 / self.t_sample_ms() * f64::from(self.n_sample())
        )?;
        writeln!(
            f,
            "prompt eval time = {:.2} ms / {} tokens ({:.2} ms per token, {:.2} tokens per second)",
            self.t_p_eval_ms(),
            self.n_p_eval(),
            self.t_p_eval_ms() / f64::from(self.n_p_eval()),
            1e3 / self.t_p_eval_ms() * f64::from(self.n_p_eval())
        )?;
        writeln!(
            f,
            "eval time = {:.2} ms / {} runs ({:.2} ms per token, {:.2} tokens per second)",
            self.t_eval_ms(),
            self.n_eval(),
            self.t_eval_ms() / f64::from(self.n_eval()),
            1e3 / self.t_eval_ms() * f64::from(self.n_eval())
        )?;
        write!(
            f,
            "total time = {:.2} ms",
            self.t_end_ms() - self.t_start_ms()
        )
    }
}
