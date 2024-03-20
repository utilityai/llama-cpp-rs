//! customized docker manifest for ollama
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ollama::OllamaError;
use crate::ollama::OLLAMA_BLOBS;

/// Represents the root structure of the Docker manifest JSON data.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Manifest {
    /// Schema version of the Docker manifest.
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,

    /// Media type of the Docker manifest.
    #[serde(rename = "mediaType")]
    pub media_type: String,

    /// Configuration object for a specific image layer.
    pub config: Config,

    /// A list of layers that make up the Docker image.
    pub layers: Vec<Layer>,
}

impl Manifest {
    /// get image model
    pub fn get_model_layer(&self) -> Result<&Layer, OllamaError> {
        const MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
        for layer in self.layers.iter() {
            if layer.media_type == MEDIA_TYPE {
                return Ok(layer);
            }
        }
        Err(OllamaError::ModelNotFound)
    }

    /// get projector model
    pub fn get_projector_layer(&self) -> Result<&Layer, OllamaError> {
        const MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
        for layer in self.layers.iter() {
            if layer.media_type == MEDIA_TYPE {
                return Ok(layer);
            }
        }
        Err(OllamaError::ModelNotFound)
    }
}

/// Represents the configuration for the Docker image specified in the manifest.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Config {
    /// Media type of the configuration object.
    #[serde(rename = "mediaType")]
    pub media_type: String,

    /// Digest of the configuration object, uniquely identifying it.
    pub digest: String,

    /// Size of the configuration object in bytes.
    pub size: u64,
}

/// Represents an individual layer within the Docker image.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Layer {
    /// Media type of the layer, indicating the format and purpose.
    #[serde(rename = "mediaType")]
    pub media_type: String,

    /// Digest of the layer, serving as a unique identifier.
    pub digest: String,

    /// Size of the layer in bytes.
    pub size: u64,
}

impl Layer {
    /// get path
    pub fn get_path(&self) -> Result<PathBuf, OllamaError> {
        let path = dirs::home_dir()
            .ok_or(OllamaError::HomeDirNotFound)?
            .join(OLLAMA_BLOBS)
            .join(&self.digest);

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests parsing of JSON into the `DockerManifest` struct.
    #[test]
    fn test_parse_json() {
        let json_data = r#"
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": "sha256:7c658f9561e5dbbafb042a00f6a4de57877adddd957809111f3123e272632b4d",
                "size": 564
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868",
                    "size": 4108916992
                },
                {
                    "mediaType": "application/vnd.ollama.image.projector",
                    "digest": "sha256:72d6f08a42f656d36b356dbe0920675899a99ce21192fd66266fb7d82ed07539",
                    "size": 624434368
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
                    "size": 11356
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": "sha256:c43332387573e98fdfad4a606171279955b53d891ba2500552c2984a6560ffb4",
                    "size": 67
                },
                {
                    "mediaType": "application/vnd.ollama.image.params",
                    "digest": "sha256:ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337",
                    "size": 30
                }
            ]
        }
        "#;

        // Attempt to deserialize the JSON data into the `DockerManifest` struct
        let parsed: Result<Manifest, serde_json::Error> = serde_json::from_str(json_data);

        // Assert that parsing is successful
        assert!(parsed.is_ok());
    }
}
