# Contribution guide

Thanks for wanting to contribute to the project!


## Updating `llama.cpp` versions

We try to keep in sync with [`llama.cpp` releases](https://github.com/ggml-org/llama.cpp/releases?q=prerelease%3Afalse).

If our version of llama.cpp is out of date relative to ^, feel free to file a PR where you:

1. Update the submodule to point to the release tag's commit.

2. Run `cargo run --bin generate-bindings` to generate the updated bindings.

3. Try to map the new APIs in `llama-cpp-sys-2/src/llama.rs` (and ideally also `llama-cpp-sys-2/src/mtmd.rs`) to rustic equivalents in `llama-cpp-2`.

4. TODO: In the future, we'll probably want to bump the version number to match, see [#1148](https://github.com/utilityai/llama-cpp-rs/issues/1148).
