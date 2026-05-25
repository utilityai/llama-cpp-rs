use crate::invalid_numa_strategy::InvalidNumaStrategy;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum NumaStrategy {
    Disabled,
    Distribute,
    Isolate,
    Numactl,
    Mirror,
}

impl TryFrom<llama_cpp_bindings_sys::ggml_numa_strategy> for NumaStrategy {
    type Error = InvalidNumaStrategy;

    fn try_from(value: llama_cpp_bindings_sys::ggml_numa_strategy) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_DISABLED => Ok(Self::Disabled),
            llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_DISTRIBUTE => Ok(Self::Distribute),
            llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_ISOLATE => Ok(Self::Isolate),
            llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_NUMACTL => Ok(Self::Numactl),
            llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_MIRROR => Ok(Self::Mirror),
            value => Err(InvalidNumaStrategy(value)),
        }
    }
}

impl From<NumaStrategy> for llama_cpp_bindings_sys::ggml_numa_strategy {
    fn from(value: NumaStrategy) -> Self {
        match value {
            NumaStrategy::Disabled => llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::Distribute => llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::Isolate => llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::Numactl => llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::Mirror => llama_cpp_bindings_sys::GGML_NUMA_STRATEGY_MIRROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NumaStrategy;
    use crate::invalid_numa_strategy::InvalidNumaStrategy;

    #[test]
    fn numa_from_and_to() {
        let numas = [
            NumaStrategy::Disabled,
            NumaStrategy::Distribute,
            NumaStrategy::Isolate,
            NumaStrategy::Numactl,
            NumaStrategy::Mirror,
        ];

        for numa in &numas {
            let raw = llama_cpp_bindings_sys::ggml_numa_strategy::from(*numa);
            let roundtripped =
                NumaStrategy::try_from(raw).expect("Failed to roundtrip NumaStrategy");

            assert_eq!(*numa, roundtripped);
        }
    }

    #[test]
    fn invalid_numa_strategy_returns_error() {
        let invalid_value = 800;
        let result = NumaStrategy::try_from(invalid_value);

        assert_eq!(result, Err(InvalidNumaStrategy(invalid_value)));
    }
}
