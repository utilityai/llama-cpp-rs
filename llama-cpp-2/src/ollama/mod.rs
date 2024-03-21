//! locate models in ollama
//!

use std::fs;

mod docker_image;
mod docker_manifest;

pub use docker_image::*;
pub use docker_manifest::*;

/// The path to the manifest file.
pub const OLLAMA_MANIFEST: &str = ".ollama/models/manifests/registry.ollama.ai/library";
/// The path to the blob directory.
pub const OLLAMA_BLOBS: &str = ".ollama/models/blobs";

/// ollama error
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    /// The model scheme is not `ollama`.
    #[error("model scheme is not `ollama`")]
    InvalidScheme,

    /// The home directory not found.
    #[error("home directory not found")]
    HomeDirNotFound,

    /// The IO error.
    #[error("{0}")]
    IoError(#[from] std::io::Error),

    /// json error
    #[error("{0}")]
    JsonError(#[from] serde_json::Error),

    /// model not found
    #[error("model not found")]
    ModelNotFound,
}

/// load a model manifest
/// # Arguments
/// - name: the name of the model, and the format is `ollama:<model_name>:<type>`. e.g. `ollama:llama2:7b`
pub fn get_model_manifest(name: &str) -> Result<Manifest, OllamaError> {
    let parts = name.split(":").collect::<Vec<_>>();
    if parts.len() != 3 || parts[0] != "ollama" {
        return Err(OllamaError::InvalidScheme);
    }
    let model_name = parts[1];
    let typ = parts[2];

    let path = dirs::home_dir()
        .ok_or(OllamaError::HomeDirNotFound)?
        .join(OLLAMA_MANIFEST)
        .join(model_name)
        .join(typ);

    // read file to string
    let json_data = fs::read_to_string(path)?;

    let manifest = serde_json::from_str::<Manifest>(&json_data)?;

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests parsing of JSON into the `DockerManifest` struct.
    #[test]
    fn test_get_model() -> Result<(), OllamaError> {
        let manifest = get_model_manifest("ollama:llava:latest");

        if let Ok(manifest) = manifest {
            let layer = manifest.get_model_layer()?;
            println!("Layer: {:?}", layer.get_path()?);

            let layer = manifest.get_projector_layer()?;
            println!("Layer: {:?}", layer.get_path()?);
        }

        Ok(())
    }
}
