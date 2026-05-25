use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use libtest_mimic::Failed;
use libtest_mimic::Trial;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::ModelSource;
use crate::llama_fixture::LlamaFixture;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::load_key::LoadKey;
use crate::phase_state::PhaseState;

fn source_label(source: ModelSource) -> String {
    match source {
        ModelSource::HuggingFace { repo, file } => format!("{repo} / {file}"),
        ModelSource::LocalPath(path) => format!("local:{path}"),
    }
}

pub struct ExecutionPhase {
    pub key: LoadKey,
    pub registrations: Vec<&'static LlamaTestRegistration>,
}

impl ExecutionPhase {
    #[must_use]
    pub fn header_line(&self, index: usize, total: usize) -> String {
        format!(
            "--- phase {phase_number}/{total_phases}: {source_label} (n_gpu_layers={n_gpu_layers}) ({trial_count} tests) ---",
            phase_number = index + 1,
            total_phases = total,
            source_label = source_label(self.key.model_source),
            n_gpu_layers = self.key.model_load_params.n_gpu_layers,
            trial_count = self.registrations.len(),
        )
    }

    pub fn print_header(&self, index: usize, total: usize) {
        eprintln!("{}", self.header_line(index, total));
    }

    pub fn run(&self, backend: &Arc<LlamaBackend>, arguments: &Arguments) -> Conclusion {
        let trials = match self.key.load_phase_state(backend) {
            Ok(state) => self.passing_trials(&Arc::new(state)),
            Err(error) => self.failing_trials(&format!("phase setup failed: {error:#}")),
        };
        libtest_mimic::run(arguments, trials)
    }

    fn passing_trials(&self, state: &Arc<PhaseState>) -> Vec<Trial> {
        self.registrations
            .iter()
            .map(|registration| {
                let state_for_trial = Arc::clone(state);
                let registration: &'static LlamaTestRegistration = registration;
                let func = registration.func;
                Trial::test(registration.name, move || {
                    let fixture = LlamaFixture {
                        model: &state_for_trial.model,
                        backend: &state_for_trial.backend,
                        context_params: &registration.context_params,
                        mtmd_context: state_for_trial.mtmd_context.as_ref(),
                        model_path: &state_for_trial.model_path,
                    };
                    func(&fixture).map_err(|error| Failed::from(format!("{error:#}")))
                })
            })
            .collect()
    }

    fn failing_trials(&self, error_message: &str) -> Vec<Trial> {
        self.registrations
            .iter()
            .map(|registration| {
                let message = error_message.to_owned();
                Trial::test(registration.name, move || Err(Failed::from(message)))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::ModelSource;
    use crate::load_key::LoadKey;
    use crate::model_load_params::ModelLoadParams;

    use super::ExecutionPhase;

    fn phase_with_source(source: ModelSource) -> ExecutionPhase {
        ExecutionPhase {
            key: LoadKey {
                model_source: source,
                mmproj_source: None,
                model_load_params: ModelLoadParams {
                    n_gpu_layers: 7,
                    use_mmap: true,
                    use_mlock: false,
                },
            },
            registrations: Vec::new(),
        }
    }

    #[test]
    fn header_line_for_huggingface_source_formats_repo_and_file() {
        let phase = phase_with_source(ModelSource::HuggingFace {
            repo: "org/name",
            file: "model.gguf",
        });

        let line = phase.header_line(0, 4);

        assert_eq!(
            line,
            "--- phase 1/4: org/name / model.gguf (n_gpu_layers=7) (0 tests) ---"
        );
    }

    #[test]
    fn header_line_for_local_path_source_uses_local_prefix() {
        let phase = phase_with_source(ModelSource::LocalPath("/abs/model.gguf"));

        let line = phase.header_line(2, 3);

        assert_eq!(
            line,
            "--- phase 3/3: local:/abs/model.gguf (n_gpu_layers=7) (0 tests) ---"
        );
    }
}
