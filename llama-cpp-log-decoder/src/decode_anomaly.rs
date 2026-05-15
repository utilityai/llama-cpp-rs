#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeAnomaly {
    OrphanCont,
    StaleBufferAbandoned,
}
