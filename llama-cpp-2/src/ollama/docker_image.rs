//! customized docker image for ollama

use serde::{Deserialize, Serialize};

/// Represents the root structure of the JSON data.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Image {
    /// The format of the model.
    pub model_format: String,

    /// The specific family of the model.
    pub model_family: String,

    /// A list of model families.
    pub model_families: Vec<String>,

    /// The type of the model.
    pub model_type: String,

    /// The type of the file.
    pub file_type: String,

    /// The architecture for which the model is intended.
    pub architecture: String,

    /// The operating system supported by the model.
    pub os: String,

    /// Filesystem information related to the model.
    pub rootfs: RootFs,
}

/// Represents the filesystem information in the `rootfs` field of the JSON data.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct RootFs {
    /// The type of the filesystem (e.g., layers, overlay).
    #[serde(rename = "type")]
    pub type_: String, // `type` is a reserved keyword in Rust, so we append an underscore

    /// The identifiers for the different filesystem layers.
    pub diff_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests parsing of JSON into the `Image` struct.
    #[test]
    fn test_parse_json() {
        let json_data = r#"
        {
            "model_format": "gguf",
            "model_family": "llama",
            "model_families": ["llama", "clip"],
            "model_type": "7B",
            "file_type": "Q4_0",
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {
                "type": "layers",
                "diff_ids": [
                    "sha256:170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868",
                    "sha256:72d6f08a42f656d36b356dbe0920675899a99ce21192fd66266fb7d82ed07539",
                    "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
                    "sha256:c43332387573e98fdfad4a606171279955b53d891ba2500552c2984a6560ffb4",
                    "sha256:ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337"
                ]
            }
        }
        "#;

        // Expected data structure, matching the JSON above.
        let expected = Image {
            model_format: "gguf".to_string(),
            model_family: "llama".to_string(),
            model_families: vec!["llama".to_string(), "clip".to_string()],
            model_type: "7B".to_string(),
            file_type: "Q4_0".to_string(),
            architecture: "amd64".to_string(),
            os: "linux".to_string(),
            rootfs: RootFs {
                type_: "layers".to_string(),
                diff_ids: vec![
                    "sha256:170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868"
                        .to_string(),
                    "sha256:72d6f08a42f656d36b356dbe0920675899a99ce21192fd66266fb7d82ed07539"
                        .to_string(),
                    "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
                        .to_string(),
                    "sha256:c43332387573e98fdfad4a606171279955b53d891ba2500552c2984a6560ffb4"
                        .to_string(),
                    "sha256:ed11eda7790d05b49395598a42b155812b17e263214292f7b87d15e14003d337"
                        .to_string(),
                ],
            },
        };

        // Deserialize the JSON data into the Rust structure
        let parsed: Result<Image, serde_json::Error> = serde_json::from_str(json_data);

        // Assert that parsing is successful and matches the expected data structure.
        assert_eq!(parsed.unwrap(), expected);
    }
}
