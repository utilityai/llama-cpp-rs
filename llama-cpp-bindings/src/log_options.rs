#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    pub disabled: bool,
    pub demote_info_to_debug: bool,
}

impl LogOptions {
    #[must_use]
    pub const fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;

        self
    }

    #[must_use]
    pub const fn with_demote_info_to_debug(mut self, demote: bool) -> Self {
        self.demote_info_to_debug = demote;

        self
    }
}

#[cfg(test)]
mod tests {
    use super::LogOptions;

    #[test]
    fn builder_chain_sets_both_flags() {
        let options = LogOptions::default()
            .with_logs_enabled(false)
            .with_demote_info_to_debug(true);

        assert!(options.disabled);
        assert!(options.demote_info_to_debug);
    }
}
