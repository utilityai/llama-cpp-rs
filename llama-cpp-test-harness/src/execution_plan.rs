//! Deterministic execution plan for the test harness.
//!
//! [`ExecutionPlan::from_registrations`] takes the registrations collected from `inventory` and
//! groups them into [`ExecutionPhase`]s by [`crate::LoadKey`]. The result is a sorted list of
//! phases — each phase corresponds to exactly one model-load cycle (load → run trials → drop).
//!
//! # Invariants
//!
//! - For every distinct [`crate::LoadKey`] the planner produces exactly one
//!   [`ExecutionPhase`]; the same key never produces two phases.
//! - Phases are sorted by [`crate::LoadKey`] (lexicographic order on the full key tuple).
//! - Registrations inside a phase are sorted by their `name`.
//! - [`crate::ContextParams`] differences within registrations sharing a key do **not** split a
//!   phase — the model loads once and each trial constructs its own `LlamaContext`.

use std::collections::BTreeMap;
use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_phase::ExecutionPhase;
use crate::llama_test_registration::LlamaTestRegistration;

fn collect_inventory_registrations() -> Vec<&'static LlamaTestRegistration> {
    inventory::iter::<LlamaTestRegistration>
        .into_iter()
        .collect()
}

pub struct ExecutionPlan {
    pub phases: Vec<ExecutionPhase>,
}

impl ExecutionPlan {
    #[must_use]
    pub fn from_registrations(registrations: &[&'static LlamaTestRegistration]) -> Self {
        let mut by_key: BTreeMap<_, Vec<&'static LlamaTestRegistration>> = BTreeMap::new();
        for registration in registrations {
            by_key
                .entry(registration.key)
                .or_default()
                .push(*registration);
        }
        let mut phases = Vec::with_capacity(by_key.len());
        for (key, mut registrations) in by_key {
            registrations.sort_by_key(|registration| registration.name);
            phases.push(ExecutionPhase { key, registrations });
        }
        Self { phases }
    }

    #[must_use]
    pub fn from_inventory() -> Self {
        let registrations = collect_inventory_registrations();
        Self::from_registrations(&registrations)
    }

    #[must_use]
    pub fn requests_void_logs(&self) -> bool {
        self.phases
            .iter()
            .any(|phase| phase.registrations.iter().any(|reg| reg.void_logs))
    }

    #[must_use]
    pub fn run(&self, backend: &Arc<LlamaBackend>, arguments: &Arguments) -> Vec<Conclusion> {
        let total = self.phases.len();
        let mut conclusions = Vec::with_capacity(total);
        for (index, phase) in self.phases.iter().enumerate() {
            phase.print_header(index, total);
            conclusions.push(phase.run(backend, arguments));
        }
        conclusions
    }
}

#[cfg(test)]
mod tests {
    use crate::context_params::ContextParams;
    use crate::llama_test_registration::LlamaTestRegistration;
    use crate::load_key::LoadKey;
    use crate::model_load_params::ModelLoadParams;
    use crate::model_source::ModelSource;
    use crate::no_op::no_op;

    use super::ExecutionPlan;

    const TRIVIAL_CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 1,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    const ALTERNATE_CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 4096,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    static REG_BETA_A: LlamaTestRegistration = LlamaTestRegistration {
        name: "alpha",
        key: LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        void_logs: false,
        func: no_op,
    };
    static REG_BETA_B: LlamaTestRegistration = LlamaTestRegistration {
        name: "bravo",
        key: LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        void_logs: false,
        func: no_op,
    };
    static REG_ALPHA_Z: LlamaTestRegistration = LlamaTestRegistration {
        name: "zulu",
        key: LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "alpha",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        void_logs: false,
        func: no_op,
    };
    static REG_BETA_DIFFERENT_CONTEXT: LlamaTestRegistration = LlamaTestRegistration {
        name: "charlie",
        key: LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        },
        context_params: ALTERNATE_CONTEXT_PARAMS,
        void_logs: false,
        func: no_op,
    };

    static REG_VOID_LOGS: LlamaTestRegistration = LlamaTestRegistration {
        name: "void-logs-trial",
        key: LoadKey {
            model_source: ModelSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        void_logs: true,
        func: no_op,
    };

    #[test]
    fn from_registrations_with_empty_input_yields_empty_plan() {
        let plan = ExecutionPlan::from_registrations(&[]);

        assert!(plan.phases.is_empty());
    }

    #[test]
    fn registrations_with_same_load_key_collapse_to_one_phase() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_BETA_B]);

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations.len(), 2);
    }

    #[test]
    fn registrations_with_distinct_load_keys_form_phases_in_load_key_sort_order() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_ALPHA_Z]);

        assert_eq!(plan.phases.len(), 2);
        assert!(matches!(
            plan.phases[0].key.model_source,
            ModelSource::HuggingFace { repo: "alpha", .. }
        ));
        assert!(matches!(
            plan.phases[1].key.model_source,
            ModelSource::HuggingFace { repo: "beta", .. }
        ));
    }

    #[test]
    fn within_a_phase_registrations_sort_by_name() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_B, &REG_BETA_A]);

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations[0].name, "alpha");
        assert_eq!(plan.phases[0].registrations[1].name, "bravo");
    }

    #[test]
    fn requests_void_logs_false_when_no_registration_opts_in() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_ALPHA_Z]);

        assert!(!plan.requests_void_logs());
    }

    #[test]
    fn requests_void_logs_true_when_any_registration_opts_in() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_VOID_LOGS]);

        assert!(plan.requests_void_logs());
    }

    #[test]
    fn registrations_sharing_a_load_key_but_differing_context_params_stay_in_one_phase() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_BETA_DIFFERENT_CONTEXT]);

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations.len(), 2);
        let context_lengths: Vec<u32> = plan.phases[0]
            .registrations
            .iter()
            .map(|registration| registration.context_params.n_ctx)
            .collect();
        assert!(context_lengths.contains(&1));
        assert!(context_lengths.contains(&4096));
    }
}
