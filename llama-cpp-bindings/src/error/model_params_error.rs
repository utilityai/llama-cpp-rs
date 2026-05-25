#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelParamsError {
    #[error("No available slot in override vector")]
    NoAvailableSlot,
    #[error("Override slot is not empty")]
    SlotNotEmpty,
    #[error("Invalid character in key: byte {byte}, {reason}")]
    InvalidCharacterInKey { byte: u8, reason: String },
}
