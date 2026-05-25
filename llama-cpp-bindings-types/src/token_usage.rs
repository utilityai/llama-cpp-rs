use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;

use serde::Deserialize;
use serde::Serialize;

use crate::token_usage_error::TokenUsageError;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub input_image_tokens: u64,
    pub input_audio_tokens: u64,
    pub content_tokens: u64,
    pub reasoning_tokens: u64,
    pub tool_call_tokens: u64,
    pub undeterminable_tokens: u64,
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
            tool_call_tokens: 0,
            undeterminable_tokens: 0,
        }
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(count);
    }

    /// # Errors
    /// Returns [`TokenUsageError::CachedExceedsPrompt`] when the running cached
    /// total would exceed [`Self::prompt_tokens`].
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

    pub const fn record_tool_call_token(&mut self) {
        self.tool_call_tokens = self.tool_call_tokens.saturating_add(1);
    }

    pub const fn record_undeterminable_token(&mut self) {
        self.undeterminable_tokens = self.undeterminable_tokens.saturating_add(1);
    }

    #[must_use]
    pub const fn completion_tokens(&self) -> u64 {
        self.content_tokens
            .saturating_add(self.reasoning_tokens)
            .saturating_add(self.tool_call_tokens)
            .saturating_add(self.undeterminable_tokens)
    }

    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens())
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
        self.tool_call_tokens = self.tool_call_tokens.saturating_add(other.tool_call_tokens);
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
    use super::TokenUsageError;

    #[test]
    fn new_starts_with_all_counters_at_zero() {
        let usage = TokenUsage::new();

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.cached_prompt_tokens, 0);
        assert_eq!(usage.input_image_tokens, 0);
        assert_eq!(usage.input_audio_tokens, 0);
        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(TokenUsage::default(), TokenUsage::new());
    }

    #[test]
    fn completion_is_zero_when_no_events_recorded() {
        assert_eq!(TokenUsage::new().completion_tokens(), 0);
    }

    #[test]
    fn total_equals_prompt_plus_completion() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(3);
        usage.record_content_token();
        usage.record_reasoning_token();

        assert_eq!(usage.total_tokens(), 5);
    }

    #[test]
    fn record_prompt_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(3);
        usage.record_prompt_tokens(4);

        assert_eq!(usage.prompt_tokens, 7);
    }

    #[test]
    fn record_cached_below_prompt_succeeds_and_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(10);
        usage
            .record_cached_prompt_tokens(3)
            .expect("3 cached <= 10 prompt is valid");
        usage
            .record_cached_prompt_tokens(4)
            .expect("3+4 cached <= 10 prompt is valid");

        assert_eq!(usage.cached_prompt_tokens, 7);
    }

    #[test]
    fn record_cached_equal_to_prompt_succeeds() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(5);
        usage
            .record_cached_prompt_tokens(5)
            .expect("5 cached == 5 prompt is valid (boundary)");

        assert_eq!(usage.cached_prompt_tokens, 5);
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
        assert_eq!(usage.cached_prompt_tokens, 0);
    }

    #[test]
    fn record_input_image_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(5);
        usage.record_input_image_tokens(3);

        assert_eq!(usage.input_image_tokens, 8);
    }

    #[test]
    fn record_input_audio_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(2);
        usage.record_input_audio_tokens(9);

        assert_eq!(usage.input_audio_tokens, 11);
    }

    #[test]
    fn input_image_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(40);

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn input_audio_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(40);

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn record_content_token_increments_only_content() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();

        assert_eq!(usage.content_tokens, 1);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_reasoning_token_increments_only_reasoning() {
        let mut usage = TokenUsage::new();
        usage.record_reasoning_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 1);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_tool_call_token_increments_only_tool_call() {
        let mut usage = TokenUsage::new();
        usage.record_tool_call_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 1);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_undeterminable_token_increments_only_undeterminable() {
        let mut usage = TokenUsage::new();
        usage.record_undeterminable_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 1);
    }

    #[test]
    fn completion_tokens_sums_every_output_kind() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();
        usage.record_content_token();
        usage.record_reasoning_token();
        usage.record_tool_call_token();
        usage.record_undeterminable_token();

        assert_eq!(usage.completion_tokens(), 5);
    }

    #[test]
    fn add_combines_field_by_field() {
        let mut left = TokenUsage::new();
        left.record_prompt_tokens(2);
        left.record_cached_prompt_tokens(1)
            .expect("1 cached <= 2 prompt is valid");
        left.record_content_token();
        left.record_reasoning_token();
        left.record_tool_call_token();
        left.record_undeterminable_token();

        let mut right = TokenUsage::new();
        right.record_prompt_tokens(5);
        right
            .record_cached_prompt_tokens(2)
            .expect("2 cached <= 5 prompt is valid");
        right.record_content_token();
        right.record_tool_call_token();

        let combined = left + right;

        assert_eq!(combined.prompt_tokens, 7);
        assert_eq!(combined.cached_prompt_tokens, 3);
        assert_eq!(combined.content_tokens, 2);
        assert_eq!(combined.reasoning_tokens, 1);
        assert_eq!(combined.tool_call_tokens, 2);
        assert_eq!(combined.undeterminable_tokens, 1);
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

        assert_eq!(combined.input_image_tokens, 7);
        assert_eq!(combined.input_audio_tokens, 8);
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

        assert_eq!(combined.prompt_tokens, 7);
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
