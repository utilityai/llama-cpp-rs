use std::path::Path;

use crate::debug_log;
use crate::library_asset_extraction::extract_lib_assets;

pub fn copy_shared_libraries(cmake_dir: &Path, target_dir: &Path) {
    let assets = extract_lib_assets(cmake_dir);

    for asset in &assets {
        let Some(filename) = asset.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        hard_link_if_missing(asset, &target_dir.join(filename));

        let examples_dir = target_dir.join("examples");

        if examples_dir.exists() {
            hard_link_if_missing(asset, &examples_dir.join(filename));
        }

        let deps_dir = target_dir.join("deps");
        hard_link_if_missing(asset, &deps_dir.join(filename));
    }
}

fn hard_link_if_missing(source: &Path, destination: &Path) {
    if destination.exists() {
        return;
    }

    debug_log!(
        "HARD LINK {} TO {}",
        source.display(),
        destination.display()
    );

    if let Err(error) = std::fs::hard_link(source, destination) {
        println!(
            "cargo:warning=failed to hard link {} to {}: {error}",
            source.display(),
            destination.display()
        );
    }
}
