# llama-cpp-rs-2

[utilityai]: https://utilityai.ca

A wrapper around the [llama-cpp](https://github.com/ggerganov/llama.cpp/) library for rust.

# Info

This is part of the project powering all the LLMs at [utilityai], it is tighly coupled llama.cpp and mimics its API as closly as possible while being safe in order to stay up to date.

# Disclaimer

This is not a simple library to use. In an ideal work a nice abstraction would be written on top of this crate to provide an ergonomic API - the benfits of this crate over raw bindings is safety and not much else.

We compensate for this shortcoming (we hope) by providing lots of examples and good documentation. Testing is a work in progress.

# Contributing

Contributions are welcome. Please open an issue before starting work on a non-trivial PR.
