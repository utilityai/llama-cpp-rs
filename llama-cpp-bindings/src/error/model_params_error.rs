/// Errors that can occur when modifying model parameters.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelParamsError {
    /// The internal override vector has no available slot.
    #[error("No available slot in override vector")]
    NoAvailableSlot,
    /// The first override slot is not empty.
    #[error("Override slot is not empty")]
    SlotNotEmpty,
    /// A character in the key is not a valid C char.
    #[error("Invalid character in key: byte {byte}, {reason}")]
    InvalidCharacterInKey {
        /// The byte value that failed conversion.
        byte: u8,
        /// The reason the conversion failed.
        reason: String,
    },
}
