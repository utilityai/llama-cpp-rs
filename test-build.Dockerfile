# Builds the project in a docker container This is used to test arm and x86 builds using github actions + docker.
# This also requires us to declare all the dependencies in the dockerfile usful as documentation.
FROM rust:bookworm AS builder
# Install requirements for bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
RUN apt update && apt install -y llvm-dev libclang-dev clang
COPY . .
RUN cargo check --package llama-cpp-sys-2