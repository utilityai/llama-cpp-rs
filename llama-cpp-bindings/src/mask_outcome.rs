use toktrie::SimpleVob;

pub enum MaskOutcome {
    Constrained(SimpleVob),
    GrammarComplete,
}
