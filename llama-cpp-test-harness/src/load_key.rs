//! Identity of one model-load operation.
//!
//! Two registrations with different [`LoadKey`]s require separate model loads. Two registrations
//! with identical [`LoadKey`]s share one load — even if every other attribute (such as
//! [`crate::ContextParams`]) differs.
//!
//! # What forces a model reload
//!
//! Only the fields of [`LoadKey`]: the model source ([`crate::ModelSource`]), the mmproj source
//! (optional [`crate::MmprojSource`]), and the [`crate::ModelLoadParams`] (`n_gpu_layers`,
//! `use_mmap`, `use_mlock`).
//!
//! # What is runtime-flexible
//!
//! Every `LlamaContextParams` setter (`n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`,
//! `n_threads_batch`, `embeddings`, and the further setters not yet surfaced in the attribute
//! schema). The harness builds a fresh `LlamaContext` per trial from `fixture.context_params`,
//! so differences here never reload the model.

use std::sync::Arc;

use anyhow::Result;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::MtmdContext;
use llama_cpp_bindings::mtmd::MtmdContextParams;

use crate::mmproj_source::MmprojSource;
use crate::model_load_params::ModelLoadParams;
use crate::model_source::ModelSource;
use crate::phase_state::PhaseState;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoadKey {
    pub model_source: ModelSource,
    pub mmproj_source: Option<MmprojSource>,
    pub model_load_params: ModelLoadParams,
}

impl LoadKey {
    /// Downloads (or resolves) the model and optional mmproj, loads them, and returns the live
    /// [`PhaseState`] that the harness keeps alive for the duration of the phase.
    ///
    /// # Errors
    ///
    /// Returns an error if any of: source resolution fails, loading the model into llama.cpp
    /// fails, or initializing the `MtmdContext` fails.
    pub fn load_phase_state(&self, backend: &Arc<LlamaBackend>) -> Result<PhaseState> {
        let model_path = self.model_source.resolve_path()?;
        let model_params = self.model_load_params.into_llama_model_params();
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)?;

        let mtmd_context = match self.mmproj_source {
            Some(mmproj_source) => {
                let mmproj_path = mmproj_source.resolve_path()?;
                let mmproj_path_str = mmproj_path.to_string_lossy();
                let params = MtmdContextParams::default();
                Some(MtmdContext::init_from_file(
                    mmproj_path_str.as_ref(),
                    &model,
                    &params,
                )?)
            }
            None => None,
        };

        Ok(PhaseState {
            mtmd_context,
            model,
            backend: Arc::clone(backend),
            model_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::mmproj_source::MmprojSource;
    use crate::model_load_params::ModelLoadParams;
    use crate::model_source::ModelSource;

    use super::LoadKey;

    fn baseline() -> LoadKey {
        LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "repo",
                file: "file",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        }
    }

    #[test]
    fn identical_keys_compare_equal() {
        assert_eq!(baseline(), baseline());
    }

    #[test]
    fn different_model_sources_compare_unequal() {
        let mut other = baseline();
        other.model_source = ModelSource::HuggingFace {
            repo: "other",
            file: "file",
        };

        assert_ne!(baseline(), other);
    }

    #[test]
    fn huggingface_and_local_path_compare_unequal() {
        let mut other = baseline();
        other.model_source = ModelSource::LocalPath("/some/local.gguf");

        assert_ne!(baseline(), other);
    }

    #[test]
    fn different_mmproj_sources_compare_unequal() {
        let mut other = baseline();
        other.mmproj_source = Some(MmprojSource::HuggingFace {
            repo: "repo",
            file: "mmproj-F16.gguf",
        });

        assert_ne!(baseline(), other);
    }

    #[test]
    fn different_model_load_params_compare_unequal() {
        let mut other = baseline();
        other.model_load_params.n_gpu_layers = 999;

        assert_ne!(baseline(), other);
    }

    // The next three tests exercise the three error-propagation paths inside
    // `load_phase_state` — model load failure, mmproj download failure, and mmproj load failure.
    // Each constructs a LoadKey whose resolution succeeds (so the path is computed) but whose
    // subsequent load step deliberately fails, then asserts the appropriate `Err` propagates.
    //
    // They share BACKEND_INIT_GATE because `LlamaBackend::init` is once-per-process.

    use std::sync::Arc;

    use llama_cpp_bindings::llama_backend::LlamaBackend;

    use crate::test_backend_gate::BACKEND_INIT_GATE;

    /// Path to the workspace `Cargo.toml`, which exists at test time but isn't a valid GGUF and
    /// isn't a valid mmproj — perfect for exercising the `load_from_file` / `init_from_file`
    /// error arms in `load_phase_state`.
    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    #[test]
    fn load_phase_state_propagates_model_load_failure() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let key = LoadKey {
            model_source: ModelSource::LocalPath(NON_GGUF_PATH),
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        };

        let result = key.load_phase_state(&backend);

        assert!(
            result.is_err(),
            "LoadKey pointing at a non-GGUF file must fail to load"
        );
    }

    #[test]
    fn load_phase_state_propagates_mmproj_download_failure() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let key = LoadKey {
            model_source: ModelSource::LocalPath(NON_GGUF_PATH),
            mmproj_source: Some(MmprojSource::HuggingFace {
                repo: "intentee-test-harness/does-not-exist",
                file: "no-such-mmproj.gguf",
            }),
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        };

        let result = key.load_phase_state(&backend);

        assert!(
            result.is_err(),
            "LoadKey with bogus mmproj HF repo must fail; the error must surface either at model \
             load (the non-GGUF model fails first) or at mmproj download"
        );
    }

    #[test]
    fn load_phase_state_propagates_mmproj_local_load_failure() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let key = LoadKey {
            model_source: ModelSource::LocalPath(NON_GGUF_PATH),
            mmproj_source: Some(MmprojSource::LocalPath(NON_GGUF_PATH)),
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        };

        let result = key.load_phase_state(&backend);

        assert!(
            result.is_err(),
            "LoadKey pointing at a non-mmproj LocalPath must fail at MtmdContext init"
        );
    }
}
