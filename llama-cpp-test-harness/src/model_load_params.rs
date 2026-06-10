use llama_cpp_bindings::model::params::LlamaModelParams;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelLoadParams {
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl ModelLoadParams {
    #[must_use]
    pub fn into_llama_model_params(self) -> LlamaModelParams {
        let Self {
            n_gpu_layers,
            use_mmap,
            use_mlock,
        } = self;
        LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mmap(use_mmap)
            .with_use_mlock(use_mlock)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelLoadParams;

    #[test]
    fn into_llama_model_params_carries_all_three_fields() {
        let params = ModelLoadParams {
            n_gpu_layers: 7,
            use_mmap: false,
            use_mlock: true,
        }
        .into_llama_model_params();

        assert_eq!(params.n_gpu_layers(), 7);
        assert!(!params.use_mmap());
        assert!(params.use_mlock());
    }

    #[test]
    fn identical_values_compare_equal() {
        let one = ModelLoadParams {
            n_gpu_layers: 1,
            use_mmap: true,
            use_mlock: false,
        };
        let two = ModelLoadParams {
            n_gpu_layers: 1,
            use_mmap: true,
            use_mlock: false,
        };

        assert_eq!(one, two);
    }

    #[test]
    fn differing_n_gpu_layers_compare_unequal() {
        let one = ModelLoadParams {
            n_gpu_layers: 1,
            use_mmap: true,
            use_mlock: false,
        };
        let two = ModelLoadParams {
            n_gpu_layers: 2,
            use_mmap: true,
            use_mlock: false,
        };

        assert_ne!(one, two);
    }
}
