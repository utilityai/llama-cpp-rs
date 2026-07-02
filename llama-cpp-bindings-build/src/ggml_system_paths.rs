use std::env;
use std::path::PathBuf;

#[derive(Debug)]
pub struct GgmlSystemPaths {
    pub cmake_dir: PathBuf,
    pub lib_dir: PathBuf,
}

impl GgmlSystemPaths {
    pub fn from_env() -> Self {
        Self {
            cmake_dir: PathBuf::from(
                env::var("DEP_GGML_CMAKE")
                    .expect("DEP_GGML_CMAKE must be provided by llama-cpp-ggml-sys"),
            ),
            lib_dir: PathBuf::from(
                env::var("DEP_GGML_LIB")
                    .expect("DEP_GGML_LIB must be provided by llama-cpp-ggml-sys"),
            ),
        }
    }
}
