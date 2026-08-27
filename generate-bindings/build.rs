//! Pass the host `cfg(target_arch = "...")` as a variable to the script.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!(
        "cargo:rustc-env=HOST_TARGET_ARCH={}",
        std::env::var("CARGO_CFG_TARGET_ARCH").unwrap()
    );
}
