#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsedModelLoadParams {
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}
