use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;

use crate::TokenUsageError;
use crate::sampled_token::SampledToken;

#[expect(
    clippy::struct_field_names,
    reason = "every field counts a kind of token"
)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenUsage {
    prompt_tokens: u64,
    cached_prompt_tokens: u64,
    input_image_tokens: u64,
    input_audio_tokens: u64,
    content_tokens: u64,
    reasoning_tokens: u64,
    undeterminable_tokens: u64,
}

impl TokenUsage {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            input_image_tokens: 0,
            input_audio_tokens: 0,
            content_tokens: 0,
            reasoning_tokens: 0,
            undeterminable_tokens: 0,
        }
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(count);
    }

    /// # Errors
    /// Returns [`TokenUsageError::CachedExceedsPrompt`] when the running cached total would
    /// exceed [`Self::prompt_tokens`].
    pub const fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), TokenUsageError> {
        let next = self.cached_prompt_tokens.saturating_add(count);

        if next > self.prompt_tokens {
            return Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: next,
                prompt: self.prompt_tokens,
            });
        }

        self.cached_prompt_tokens = next;

        Ok(())
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.input_image_tokens = self.input_image_tokens.saturating_add(count);
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.input_audio_tokens = self.input_audio_tokens.saturating_add(count);
    }

    pub const fn record_content_token(&mut self) {
        self.content_tokens = self.content_tokens.saturating_add(1);
    }

    pub const fn record_reasoning_token(&mut self) {
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
    }

    pub const fn record_undeterminable_token(&mut self) {
        self.undeterminable_tokens = self.undeterminable_tokens.saturating_add(1);
    }

    pub const fn record_sampled(&mut self, token: &SampledToken) {
        match token {
            SampledToken::Content(_) => self.record_content_token(),
            SampledToken::Reasoning(_) => self.record_reasoning_token(),
            SampledToken::Undeterminable(_) => self.record_undeterminable_token(),
        }
    }

    #[must_use]
    pub const fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens
    }

    #[must_use]
    pub const fn cached_prompt_tokens(&self) -> u64 {
        self.cached_prompt_tokens
    }

    #[must_use]
    pub const fn input_image_tokens(&self) -> u64 {
        self.input_image_tokens
    }

    #[must_use]
    pub const fn input_audio_tokens(&self) -> u64 {
        self.input_audio_tokens
    }

    #[must_use]
    pub const fn content_tokens(&self) -> u64 {
        self.content_tokens
    }

    #[must_use]
    pub const fn reasoning_tokens(&self) -> u64 {
        self.reasoning_tokens
    }

    #[must_use]
    pub const fn undeterminable_tokens(&self) -> u64 {
        self.undeterminable_tokens
    }

    #[must_use]
    pub const fn completion_tokens(&self) -> u64 {
        self.content_tokens.saturating_add(self.reasoning_tokens)
    }
}

impl Add for TokenUsage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;

        self
    }
}

impl Add<&Self> for TokenUsage {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self += other;

        self
    }
}

impl AddAssign for TokenUsage {
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl AddAssign<&Self> for TokenUsage {
    fn add_assign(&mut self, other: &Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.cached_prompt_tokens = self
            .cached_prompt_tokens
            .saturating_add(other.cached_prompt_tokens);
        self.input_image_tokens = self
            .input_image_tokens
            .saturating_add(other.input_image_tokens);
        self.input_audio_tokens = self
            .input_audio_tokens
            .saturating_add(other.input_audio_tokens);
        self.content_tokens = self.content_tokens.saturating_add(other.content_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        self.undeterminable_tokens = self
            .undeterminable_tokens
            .saturating_add(other.undeterminable_tokens);
    }
}

impl Sum for TokenUsage {
    fn sum<TIter: Iterator<Item = Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), |acc, item| acc + item)
    }
}

impl<'usage> Sum<&'usage Self> for TokenUsage {
    fn sum<TIter: Iterator<Item = &'usage Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), |acc, item| acc + item)
    }
}

#[cfg(test)]
mod tests {
    use super::TokenUsage;
    use crate::TokenUsageError;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    const TOKEN: LlamaToken = LlamaToken::new(7);

    #[test]
    fn new_starts_with_all_counters_at_zero() {
        let usage = TokenUsage::new();

        assert_eq!(usage.prompt_tokens(), 0);
        assert_eq!(usage.cached_prompt_tokens(), 0);
        assert_eq!(usage.input_image_tokens(), 0);
        assert_eq!(usage.input_audio_tokens(), 0);
        assert_eq!(usage.content_tokens(), 0);
        assert_eq!(usage.reasoning_tokens(), 0);
        assert_eq!(usage.undeterminable_tokens(), 0);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(TokenUsage::default(), TokenUsage::new());
    }

    #[test]
    fn completion_is_zero_when_no_events_recorded() {
        let usage = TokenUsage::new();

        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn record_prompt_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(3);
        usage.record_prompt_tokens(4);

        assert_eq!(usage.prompt_tokens(), 7);
    }

    #[test]
    fn record_cached_below_prompt_succeeds_and_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(10);
        usage.record_cached_prompt_tokens(3).unwrap();
        usage.record_cached_prompt_tokens(4).unwrap();

        assert_eq!(usage.cached_prompt_tokens(), 7);
    }

    #[test]
    fn record_cached_equal_to_prompt_succeeds() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(5);
        usage.record_cached_prompt_tokens(5).unwrap();

        assert_eq!(usage.cached_prompt_tokens(), 5);
    }

    #[test]
    fn record_cached_above_prompt_returns_error_and_does_not_mutate() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(2);

        let result = usage.record_cached_prompt_tokens(3);

        assert_eq!(
            result,
            Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: 3,
                prompt: 2,
            })
        );
        assert_eq!(usage.cached_prompt_tokens(), 0);
    }

    #[test]
    fn record_cached_can_be_recorded_after_more_prompt_tokens_arrive() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(2);

        let first = usage.record_cached_prompt_tokens(3);
        assert!(first.is_err());

        usage.record_prompt_tokens(5);
        usage.record_cached_prompt_tokens(3).unwrap();

        assert_eq!(usage.cached_prompt_tokens(), 3);
    }

    #[test]
    fn record_input_image_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(5);
        usage.record_input_image_tokens(3);

        assert_eq!(usage.input_image_tokens(), 8);
    }

    #[test]
    fn record_input_audio_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(2);
        usage.record_input_audio_tokens(9);

        assert_eq!(usage.input_audio_tokens(), 11);
    }

    #[test]
    fn input_image_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(40);

        assert_eq!(usage.prompt_tokens(), 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn input_audio_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(40);

        assert_eq!(usage.prompt_tokens(), 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn record_sampled_content_increments_only_content() {
        let mut usage = TokenUsage::new();
        usage.record_sampled(&SampledToken::Content(TOKEN));

        assert_eq!(usage.content_tokens(), 1);
        assert_eq!(usage.reasoning_tokens(), 0);
        assert_eq!(usage.undeterminable_tokens(), 0);
    }

    #[test]
    fn record_sampled_reasoning_increments_only_reasoning() {
        let mut usage = TokenUsage::new();
        usage.record_sampled(&SampledToken::Reasoning(TOKEN));

        assert_eq!(usage.content_tokens(), 0);
        assert_eq!(usage.reasoning_tokens(), 1);
        assert_eq!(usage.undeterminable_tokens(), 0);
    }

    #[test]
    fn record_sampled_undeterminable_increments_only_undeterminable() {
        let mut usage = TokenUsage::new();
        usage.record_sampled(&SampledToken::Undeterminable(TOKEN));

        assert_eq!(usage.content_tokens(), 0);
        assert_eq!(usage.reasoning_tokens(), 0);
        assert_eq!(usage.undeterminable_tokens(), 1);
    }

    #[test]
    fn undeterminable_tokens_do_not_contribute_to_completion_tokens() {
        let mut usage = TokenUsage::new();
        usage.record_undeterminable_token();
        usage.record_undeterminable_token();

        assert_eq!(usage.undeterminable_tokens(), 2);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn completion_tokens_sums_only_content_and_reasoning() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();
        usage.record_content_token();
        usage.record_reasoning_token();

        assert_eq!(usage.completion_tokens(), 3);
    }

    #[test]
    fn independent_instances_do_not_share_counts() {
        let mut first = TokenUsage::new();
        let mut second = TokenUsage::new();

        first.record_prompt_tokens(11);
        first.record_content_token();

        second.record_reasoning_token();

        assert_eq!(first.prompt_tokens(), 11);
        assert_eq!(first.content_tokens(), 1);
        assert_eq!(first.reasoning_tokens(), 0);

        assert_eq!(second.prompt_tokens(), 0);
        assert_eq!(second.content_tokens(), 0);
        assert_eq!(second.reasoning_tokens(), 1);
    }

    #[test]
    fn add_combines_field_by_field() {
        let mut left = TokenUsage::new();
        left.record_prompt_tokens(2);
        left.record_cached_prompt_tokens(1).unwrap();
        left.record_content_token();
        left.record_reasoning_token();
        left.record_undeterminable_token();

        let mut right = TokenUsage::new();
        right.record_prompt_tokens(5);
        right.record_cached_prompt_tokens(2).unwrap();
        right.record_content_token();

        let combined = left + right;

        assert_eq!(combined.prompt_tokens(), 7);
        assert_eq!(combined.cached_prompt_tokens(), 3);
        assert_eq!(combined.content_tokens(), 2);
        assert_eq!(combined.reasoning_tokens(), 1);
        assert_eq!(combined.undeterminable_tokens(), 1);
    }

    #[test]
    fn add_combines_image_and_audio_fields_too() {
        let mut left = TokenUsage::new();
        left.record_input_image_tokens(3);
        left.record_input_audio_tokens(7);

        let mut right = TokenUsage::new();
        right.record_input_image_tokens(4);
        right.record_input_audio_tokens(1);

        let combined = left + right;

        assert_eq!(combined.input_image_tokens(), 7);
        assert_eq!(combined.input_audio_tokens(), 8);
    }

    #[test]
    fn add_assign_matches_add() {
        let mut left = TokenUsage::new();
        left.record_prompt_tokens(2);
        left.record_content_token();

        let mut right = TokenUsage::new();
        right.record_prompt_tokens(3);
        right.record_reasoning_token();

        let added = left + right;

        let mut accumulating = left;
        accumulating += right;

        assert_eq!(accumulating, added);
    }

    #[test]
    fn add_ref_combines_field_by_field() {
        let mut left = TokenUsage::new();
        left.record_prompt_tokens(2);

        let mut right = TokenUsage::new();
        right.record_prompt_tokens(5);
        let right_ref = &right;

        let combined = left + right_ref;

        assert_eq!(combined.prompt_tokens(), 7);
    }

    #[test]
    fn sum_over_iter_matches_repeated_add_assign() {
        let mut a = TokenUsage::new();
        a.record_prompt_tokens(1);

        let mut b = TokenUsage::new();
        b.record_prompt_tokens(2);
        b.record_content_token();

        let mut c = TokenUsage::new();
        c.record_prompt_tokens(4);
        c.record_reasoning_token();

        let summed: TokenUsage = [a, b, c].into_iter().sum();
        let summed_ref: TokenUsage = [&a, &b, &c].into_iter().sum();

        let mut acc = TokenUsage::new();
        acc += a;
        acc += b;
        acc += c;

        assert_eq!(summed, acc);
        assert_eq!(summed_ref, acc);
    }

    #[test]
    fn sum_over_empty_iter_returns_default() {
        let summed: TokenUsage = std::iter::empty::<TokenUsage>().sum();

        assert_eq!(summed, TokenUsage::default());
    }
}
