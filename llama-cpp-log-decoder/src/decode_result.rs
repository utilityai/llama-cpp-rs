use crate::decode_anomaly::DecodeAnomaly;
use crate::decode_output::DecodeOutput;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeResult {
    pub output: DecodeOutput,
    pub anomaly: Option<DecodeAnomaly>,
}
