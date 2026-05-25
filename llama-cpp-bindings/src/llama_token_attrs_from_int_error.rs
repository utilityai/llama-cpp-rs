#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenAttrsFromIntError {
    #[error("Unknown Value {0}")]
    UnknownValue(std::ffi::c_uint),
}
