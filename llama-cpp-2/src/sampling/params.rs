//! Safe parameters used to construct [`super::LlamaSampler`]

/// Safe parameters used to construct [`super::LlamaSampler`]
#[derive(Debug, Clone, Copy)]
pub enum LlamaSamplerParams<'a> {
    /// A chain of samplers, applied one after another
    #[allow(missing_docs)]
    Chain {
        no_perf: bool,
        stages: &'a [LlamaSamplerParams<'a>],
    },

    /// Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original
    /// value, the rest are set to -inf
    Temp(f32),

    /// Dynamic temperature implementation (a.k.a. entropy) described in the paper <https://arxiv.org/abs/2309.02772>.
    #[allow(missing_docs)]
    TempExt { t: f32, delta: f32, exponent: f32 },
    /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>
    TopK(i32),
    /// Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
    #[allow(missing_docs)]
    Typical { p: f32, min_keep: usize },
    /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>
    #[allow(missing_docs)]
    TopP { p: f32, min_keep: usize },
    /// Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
    #[allow(missing_docs)]
    MinP { p: f32, min_keep: usize },

    /// XTC sampler as described in <https://github.com/oobabooga/text-generation-webui/pull/6335>
    #[allow(missing_docs)]
    Xtc {
        /// The probability of this sampler being applied.
        p: f32,
        t: f32,
        min_keep: usize,
        /// Seed to use when selecting whether to apply this sampler or not
        seed: u32,
    },

    /// Grammar sampler
    #[allow(missing_docs)]
    Grammar {
        model: &'a crate::model::LlamaModel,
        string: &'a str,
        root: &'a str,
    },

    ///  @details DRY sampler, designed by p-e-w, as described in:
    ///  <https://github.com/oobabooga/text-generation-webui/pull/5677>, porting Koboldcpp
    ///  implementation authored by pi6am: <https://github.com/LostRuins/koboldcpp/pull/982>
    #[allow(missing_docs)]
    Dry {
        model: &'a crate::model::LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: &'a [&'a str],
    },

    /// Penalizes tokens for being present in the context.
    Penalties {
        /// ``model.n_vocab()``
        n_vocab: i32,
        /// ``model.token_eos()``
        special_eos_id: i32,
        /// ``model.token_nl()``
        linefeed_id: i32,
        /// last n tokens to penalize (0 = disable penalty, -1 = context size)
        penalty_last_n: i32,
        /// 1.0 = disabled
        penalty_repeat: f32,
        /// 0.0 = disabled
        penalty_freq: f32,
        /// 0.0 = disabled
        penalty_present: f32,
        /// consider newlines as a repeatable token
        penalize_nl: bool,
        /// ignore the end-of-sequence token
        ignore_eos: bool,
    },

    /// Select a token at random based on each token's probabilities
    Dist {
        /// Seed to initialize random generation with
        seed: u32,
    },

    /// Select the most likely token
    Greedy,
}

impl<'a> LlamaSamplerParams<'a> {
    /// Easily create a chain of samplers with performance metrics enabled.
    #[must_use]
    pub fn chain(stages: &'a [Self]) -> Self {
        LlamaSamplerParams::Chain {
            no_perf: false,
            stages,
        }
    }

    /// Easily create a [`LlamaSamplerParams::Penalties`] sampler using a model. This sets
    /// `penalize_nl` to false and `ignore_eos` to true as reasonable defaults.
    #[must_use]
    pub fn penalties(
        model: &'a crate::model::LlamaModel,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        Self::Penalties {
            n_vocab: model.n_vocab(),
            special_eos_id: model.token_eos().0,
            linefeed_id: model.token_nl().0,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            penalize_nl: false,
            ignore_eos: true,
        }
    }

    /// Easily define a [`LlamaSamplerParams::Typical`] with `min_keep == 1`
    #[must_use]
    pub fn typical(p: f32) -> Self {
        Self::Typical { p, min_keep: 1 }
    }

    /// Easily define a [`LlamaSamplerParams::TopP`] with `min_keep == 1`
    #[must_use]
    pub fn top_p(p: f32) -> Self {
        Self::TopP { p, min_keep: 1 }
    }

    /// Easily define a [`LlamaSamplerParams::MinP`] with `min_keep == 1`
    #[must_use]
    pub fn min_p(p: f32) -> Self {
        Self::MinP { p, min_keep: 1 }
    }

    /// Whether this sampler's outputs are dependent on the tokens in the model's context. 
    pub(crate) fn uses_context_tokens(&self) -> bool {
        match self {
            LlamaSamplerParams::Chain { stages, .. } => {
                stages.iter().any(LlamaSamplerParams::uses_context_tokens)
            }

            LlamaSamplerParams::Grammar { .. }
            | LlamaSamplerParams::Penalties { .. }
            | LlamaSamplerParams::Dry { .. } => true,

            LlamaSamplerParams::Temp(_)
            | LlamaSamplerParams::TempExt { .. }
            | LlamaSamplerParams::TopK(_)
            | LlamaSamplerParams::Typical { .. }
            | LlamaSamplerParams::TopP { .. }
            | LlamaSamplerParams::MinP { .. }
            | LlamaSamplerParams::Xtc { .. }
            | LlamaSamplerParams::Dist { .. }
            | LlamaSamplerParams::Greedy => false,
        }
    }
}
